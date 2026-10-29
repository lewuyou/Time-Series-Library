# -*- coding: utf-8 -*-
"""命中率分析：预测涨幅阈值切分 / 每日预测值分位数切分。

输入：checkpoints/<run>/test_predictions.csv（train_global_timexer.py 输出）
      dataset/global_002/universe.csv（股票名称/行业）

模式：
  threshold（默认）：预测涨幅 > T%（如 0/1/2/3/5%）时实际上涨命中率。
  quantile         ：每日按 y_pred 降序取前 q%（如 1/5/10/20%）的信号股，
                     统计"实际是否上涨"命中率；对主分位输出个股 Top 榜。

口径：
    test 集为导出时保留的最后 30 个交易日。
    命中   : 信号日 y_true > 0
    命中率 = 命中数 / 信号数；基准 = 全体实际上涨占比
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="checkpoints/global_final_001")
    p.add_argument("--mode", choices=["threshold", "quantile"], default="threshold")
    # threshold 模式
    p.add_argument("--thresholds-pct", default="0,1,2,3,5",
                   help="逗号分隔的百分比阈值")
    # quantile 模式
    p.add_argument("--quantiles-pct", default="1,5,10,20",
                   help="逗号分隔的每日分位数（%）")
    p.add_argument("--main-quantile-pct", type=float, default=5,
                   help="输出个股 Top 榜的主分位数")
    p.add_argument("--floor-pct", type=float, default=0,
                   help="预测涨幅绝对下限（%%），混合规则：排名前 q%% 且 y_pred > 下限")
    # 公共
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-days", type=int, default=2,
                   help="个股进入 Top 榜所需的最少信号天数")
    return p.parse_args()


def load_frame(run: Path) -> pd.DataFrame:
    frame = pd.read_csv(run / "test_predictions.csv", dtype={"symbol": str})
    universe = pd.read_csv(
        "dataset/global_002/universe.csv",
        dtype={"symbol": str}, encoding="utf-8-sig",
    )[["symbol", "name", "industry_l2", "total_mv", "mv_rank"]]
    return frame.merge(universe, on="symbol", how="left")


def run_threshold(args, run, frame):
    thresholds = [float(x) / 100 for x in args.thresholds_pct.split(",")]
    base_rate = (frame.y_true > 0).mean()

    print("== 全池汇总（%d 只 × 30 个交易日）==" % frame.symbol.nunique())
    print(f"全体实际上涨占比（基准）: {base_rate:.1%}")
    print(f"{'阈值':>6} {'预测天数':>8} {'命中(实际涨)':>10} {'命中率':>8} {'达标日实际均涨幅':>12}")
    for th in thresholds:
        mask = frame.y_pred > th
        n = int(mask.sum())
        hit = int((frame.y_true[mask] > 0).sum())
        prec = hit / n if n else np.nan
        avg = frame.y_true[mask].mean() * 100 if n else np.nan
        tag = f"{th * 100:g}%"
        print(f"{tag:>6} {n:>8d} {hit:>10d} "
              f"{(f'{prec:.1%}' if n else '-'):>8} "
              f"{(f'{avg:.2f}%' if n else '-'):>12}")

    frame["up"] = frame.y_true > 0
    records = []
    for symbol, part in frame.groupby("symbol"):
        meta = part.iloc[0]
        row = {
            "symbol": symbol, "name": meta["name"],
            "industry_l2": meta["industry_l2"], "test_days": len(part),
            "actual_up_days": int(part.up.sum()),
            "dir_accuracy": float((np.sign(part.y_pred) == np.sign(part.y_true)).mean()),
            "cum_true_return_pct": float(((1 + part.y_true).prod() - 1) * 100),
        }
        for th in thresholds:
            mask = part.y_pred > th
            n = int(mask.sum())
            hit = int((part.y_true[mask] > 0).sum())
            tag = f"{th * 100:g}"
            row[f"n@{tag}pct"], row[f"hit@{tag}pct"] = n, hit
            row[f"prec@{tag}pct"] = hit / n if n else np.nan
        records.append(row)
    table = pd.DataFrame(records)
    table.to_csv(run / "hit_rate_by_stock.csv", index=False)

    for th in thresholds:
        tag = f"{th * 100:g}"
        ncol, hcol, pcol = f"n@{tag}pct", f"hit@{tag}pct", f"prec@{tag}pct"
        qualified = table[table[ncol] >= args.min_days].copy()
        print(f"\n== 预测涨幅 > {tag}% 的 Top {args.top}（要求达标天数 >= {args.min_days}，"
              f"合格 {len(qualified)} 只）==")
        if qualified.empty:
            print("  无个股满足最低达标天数。")
            continue
        top = qualified.sort_values([pcol, ncol], ascending=False).head(args.top)
        top.to_csv(run / f"top10_stocks_th{tag}pct.csv", index=False)
        show = top[["symbol", "name", "industry_l2", ncol, hcol, pcol,
                    "cum_true_return_pct"]].copy()
        show[pcol] = (show[pcol] * 100).round(1)
        show.columns = ["symbol", "name", "industry", "pred_days", "hit_days",
                        "hit_rate%", "cum_true_return%"]
        show["cum_true_return%"] = show["cum_true_return%"].round(2)
        print(show.to_string(index=False))


def signal_mask_by_day(part: pd.DataFrame, q: float) -> pd.Series:
    """part: 单日数据；返回 bool Series，y_pred 降序前 ceil(n*q) 名为 True。"""
    k = max(1, int(np.ceil(len(part) * q)))
    return part.y_pred.rank(ascending=False, method="first") <= k


def run_quantile(args, run, frame):
    quantiles = [float(x) / 100 for x in args.quantiles_pct.split(",")]
    main_q = args.main_quantile_pct / 100
    base_rate = (frame.y_true > 0).mean()
    n_days = frame.target_date.nunique()

    floor = args.floor_pct / 100
    floor_tag = f" 且预测 > {args.floor_pct:g}%" if floor > 0 else ""
    print(f"== 全池：每日按预测值取前 q%{floor_tag} 的信号命中率（{n_days} 天）==")
    print(f"全体实际上涨占比（基准）: {base_rate:.1%}")
    print(f"{'分位':>6} {'日均信号数':>8} {'信号总数':>8} {'命中':>6} {'命中率':>8} {'信号日实际均涨幅':>12}")

    frame = frame.copy()
    frame["day_rank"] = frame.groupby("target_date")["y_pred"].rank(
        ascending=False, method="first")
    day_n = frame.groupby("target_date")["y_pred"].transform("size")

    main_sig = None
    for q in quantiles:
        k = np.ceil(day_n * q).clip(lower=1)
        sig = frame[(frame.day_rank <= k) & (frame.y_pred > floor)]
        n, hit = len(sig), int((sig.y_true > 0).sum())
        prec = hit / n if n else np.nan
        avg = sig.y_true.mean() * 100 if n else np.nan
        per_day = len(sig) / n_days
        tag = f"{q * 100:g}%"
        print(f"{tag:>6} {per_day:>8.1f} {n:>8d} {hit:>6d} "
              f"{(f'{prec:.1%}' if n else '-'):>8} "
              f"{(f'{avg:.2f}%' if n else '-'):>12}")
        if abs(q - main_q) < 1e-9:
            main_sig = sig.copy()

    if main_sig is None:
        raise SystemExit(f"--main-quantile-pct {args.main_quantile_pct} 不在 --quantiles-pct 列表中")

    # ---- 主分位：逐日命中率 ----
    daily = main_sig.groupby("target_date").agg(
        signals=("y_true", "size"), hits=("y_true", lambda s: int((s > 0).sum())))
    daily["hit_rate"] = daily.hits / daily.signals
    day_base = frame.groupby("target_date").y_true.apply(lambda s: (s > 0).mean())
    daily["day_up_base"] = day_base
    suffix = "" if floor <= 0 else f"_floor{args.floor_pct:g}pct"
    daily.to_csv(run / f"quantile_daily_hit{suffix}.csv")
    win_days = int((daily.hit_rate > daily.day_up_base).sum())
    print(f"\n主分位 {args.main_quantile_pct:g}%{floor_tag}："
          f"有信号的天数 {len(daily)}/{n_days}，"
          f"其中逐日命中率跑赢当日全池上涨占比 {win_days}/{len(daily)} 天")

    # ---- 主分位：个股榜 ----
    main_sig["up"] = main_sig.y_true > 0
    records = []
    for symbol, part in main_sig.groupby("symbol"):
        meta = part.iloc[0]
        records.append({
            "symbol": symbol, "name": meta["name"], "industry_l2": meta["industry_l2"],
            "signal_days": len(part), "hit_days": int(part.up.sum()),
            "precision": part.up.mean(),
            "avg_true_return_pct": part.y_true.mean() * 100,
            "avg_pred_return_pct": part.y_pred.mean() * 100,
        })
    table = pd.DataFrame(records).sort_values(
        ["precision", "signal_days"], ascending=False)
    table.to_csv(run / f"quantile_stocks_q{args.main_quantile_pct:g}pct{suffix}.csv",
                 index=False)

    qualified = table[table.signal_days >= args.min_days]
    print(f"\n== 每日前 {args.main_quantile_pct:g}%{floor_tag} 信号的个股 Top {args.top}"
          f"（信号天数 >= {args.min_days}，合格 {len(qualified)}/{len(table)} 只在榜）==")
    if qualified.empty:
        print("  无个股满足最低信号天数。")
        return
    show = qualified.head(args.top).copy()
    show["precision"] = (show.precision * 100).round(1)
    for col in ("avg_true_return_pct", "avg_pred_return_pct"):
        show[col] = show[col].round(2)
    show.columns = ["symbol", "name", "industry", "signal_days", "hit_days",
                    "hit_rate%", "avg_true_return%", "avg_pred_return%"]
    print(show.to_string(index=False))


def main():
    args = parse_args()
    run = Path(args.run_dir)
    frame = load_frame(run)
    if args.mode == "threshold":
        run_threshold(args, run, frame)
    else:
        run_quantile(args, run, frame)


if __name__ == "__main__":
    main()
