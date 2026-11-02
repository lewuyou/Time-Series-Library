# -*- coding: utf-8 -*-
"""按混合规则评估测试期逐日选股正收益命中率。

自 analyze_daily_hit.py 原样迁入，逐日排名/信号判定改用共享的
stock.selection.rules.hybrid_signal（floor 单位换算为小数，与
y_pred 同尺度），统计口径与原实现逐位一致。

规则（与每日预测完全一致）：
    每个交易日按 y_pred（原始收益率尺度）降序排名，
    信号 = 排名前 q%（ceil） 且 预测涨幅 > floor%（默认 0.5%）；
    命中 = 当日信号股的 y_true > 0。

输入：checkpoints/<run>/test_predictions.csv（最后 N 个交易日逐样本预测）
输出（写入 <run>/）：
    quantile_daily_hit.csv            逐日：信号数/命中数/命中率/当日全池上涨占比
    quantile_daily_hit_floor*.csv     其他 floor 下限的对照
    quantile_stocks_q*.csv            每只股票的信号天数/命中天数/精度
    daily_hit_summary.json            汇总：整体命中率、基准上涨占比、提升

用法：
    python -m stock.evaluation.daily_hit --run-dir checkpoints/global_001
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock.selection.rules import hybrid_signal
from stock.utils import write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--quantile-pct", type=float, default=5.0)
    p.add_argument("--floors", type=float, nargs="*", default=[0.2, 0.3, 0.5])
    return p.parse_args()


def daily_hit(frame, q_pct, floor_pct):
    rows = []
    for date, part in frame.groupby("target_date"):
        # 分位截断逐日计算（当日股票数 n），不能跨日池化排名
        # 命中分析口径：y_pred 为小数收益率，floor 0.5% 换算为 0.5/100；
        # 与生产链路（百分数 pred + floor_pct=0.5）数值等价、单位不同。
        signal = hybrid_signal(
            part["y_pred"], q_pct,
            floor_decimal=floor_pct / 100.0, pred_unit="decimal",
            symbol=part["symbol"], eligible=part["symbol"].tolist(),
        )["signal"]
        sig = part[signal]
        hits = (sig.y_true > 0).sum() if len(sig) else 0
        rows.append({
            "target_date": date,
            "signals": int(signal.sum()),
            "hits": int(hits),
            "hit_rate": hits / len(sig) if len(sig) else np.nan,
            "day_up_base": (part.y_true > 0).mean(),
            "day_mean_true_ret_pct": part.y_true.mean() * 100,
            "signal_mean_true_ret_pct": (
                sig.y_true.mean() * 100 if len(sig) else np.nan
            ),
        })
    daily = pd.DataFrame(rows)
    pooled_signals = daily.signals.sum()
    pooled_hits = daily.hits.sum()
    # 信号组合整体平均真实收益率（逐日等权信号股）
    sig_returns = [
        r["signal_mean_true_ret_pct"]
        for _, r in daily.iterrows() if np.isfinite(r["signal_mean_true_ret_pct"])
    ]
    return daily, {
        "days": len(daily),
        "quantile_pct": q_pct,
        "floor_pct": floor_pct,
        "total_signals": int(pooled_signals),
        "total_hits": int(pooled_hits),
        "pooled_hit_rate": (
            pooled_hits / pooled_signals if pooled_signals else None
        ),
        "base_up_rate": float(frame.y_true.gt(0).mean()),
        "avg_daily_hit_rate": float(daily.hit_rate.mean()),
        "signal_days": int((daily.signals > 0).sum()),
        "signal_mean_true_ret_pct": (
            float(np.mean(sig_returns)) if sig_returns else None
        ),
        "all_mean_true_ret_pct": float(frame.y_true.mean() * 100),
        "beat_base": (
            (pooled_hits / pooled_signals) - frame.y_true.gt(0).mean()
            if pooled_signals else None
        ),
    }


def per_stock(frame, q_pct, floor_pct):
    rows = []
    for date, part in frame.groupby("target_date"):
        # 命中分析口径：y_pred 为小数收益率，floor 0.5% 换算为 0.5/100；
        # 与生产链路（百分数 pred + floor_pct=0.5）数值等价、单位不同。
        signal = hybrid_signal(
            part["y_pred"], q_pct,
            floor_decimal=floor_pct / 100.0, pred_unit="decimal",
            symbol=part["symbol"], eligible=part["symbol"].tolist(),
        )["signal"]
        rows.append(part[signal].assign(target_date=date))
    sig_all = pd.concat(rows) if rows else frame.iloc[0:0]
    if not len(sig_all):
        return pd.DataFrame()
    table = sig_all.groupby("symbol").agg(
        signal_days=("y_true", "size"),
        hit_days=("y_true", lambda s: int((s > 0).sum())),
        avg_true_return_pct=("y_true", lambda s: s.mean() * 100),
        avg_pred_return_pct=("y_pred", lambda s: s.mean() * 100),
    ).reset_index()
    table["precision"] = table.hit_days / table.signal_days
    return table.sort_values(["precision", "signal_days"], ascending=False)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    frame = pd.read_csv(run / "test_predictions.csv", dtype={"symbol": str})

    summary = {}
    for floor in args.floors:
        daily, stats = daily_hit(frame, args.quantile_pct, floor)
        tag = f"floor{floor:g}pct"
        if floor == 0.5:
            daily.to_csv(run / "quantile_daily_hit.csv", index=False)
        daily.to_csv(run / f"quantile_daily_hit_{tag}.csv", index=False)
        summary[f"q{args.quantile_pct:g}_floor{floor:g}"] = stats

    # 信号股逐股明细（主 floor=0.5）
    stocks = per_stock(frame, args.quantile_pct, 0.5)
    if len(stocks):
        stocks.to_csv(
            run / f"quantile_stocks_q{args.quantile_pct:g}pct.csv", index=False
        )

    # 参考口径：全部"预测>0"的命中率（无排名阈值）
    pred_up = frame[frame.y_pred > 0]
    summary["all_pred_up"] = {
        "total": int(len(pred_up)),
        "hit_rate": float((pred_up.y_true > 0).mean()),
    }
    summary["direction_accuracy_nonzero"] = float(
        (np.sign(frame.y_pred) == np.sign(frame.y_true))[
            frame.y_true != 0
        ].mean()
    )

    write_json(run / "daily_hit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    main_daily = pd.read_csv(run / "quantile_daily_hit.csv")
    print("\n逐日命中率（floor=0.5%）：")
    print(main_daily.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
