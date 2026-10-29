# -*- coding: utf-8 -*-
"""每日选股预测：共享 TimeXer 模型全池打分 + 混合规则选股（第 4 步生产化）。

流程（每个交易日收盘后运行，运行前请先更新 AutoTS 下载库）：
    1. 对 checkpoint 登记的每只股票调用 dataset.py --mode infer，
       用冻结的训练期预处理参数从 duckdb 生成最新推理窗口 latest.npz；
    2. 批量过模型得到下一交易日简单收益率预测（还原到原始尺度）；
    3. 混合规则选股（与回测口径一致）：
           当日预测值排名前 q%（默认 5%） 且 预测涨幅 > 下限（默认 0.5%）；
    4. 输出排名榜与信号股，保存 CSV。

用法：
    # 完整流程（infer 导出 + 打分）
    python predict_daily.py \
        --checkpoint checkpoints/global_final_001/checkpoint.pth \
        --data-root dataset/global_002 \
        --db ../AutoTS/download/autots.duckdb \
        --infer-root dataset/global_infer_001 \
        --workers 4 --top 10

    # 只重打分（复用已有 infer 导出，跳过步骤 1）
    python predict_daily.py --checkpoint ... --data-root ... --skip-export

输出：
    终端：数据新鲜度 / 打分统计 / Top 榜（标注是否通过混合规则）
    predictions/predict_<target_date>.csv   全池排名明细

注意：
    - 预测的是 t+1 收盘相对 t 收盘的后复权简单收益率；信号在 t+1 早
      08:30 才能形成，无法按 t 收盘价成交，实际可交易价格见开盘。
    - 停牌/数据滞后的股票（input_date 落后于全池最新日）不参与信号
      排名，但在 CSV 中保留并标注。
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_global_timexer import (
    TARGET, align_features, build_model, compute_ex_mask,
    exogenous_mask_layout, read_json,
)

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True,
                   help="训练期导出目录（读 preprocessing.json / universe.csv）")
    p.add_argument("--db", default="../AutoTS/download/autots.duckdb")
    p.add_argument("--infer-root", default="dataset/global_infer_001")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--quantile-pct", type=float, default=5,
                   help="混合规则：当日预测排名前 q%%")
    p.add_argument("--floor-pct", type=float, default=0.5,
                   help="混合规则：预测涨幅绝对下限（%%）")
    p.add_argument("--top", type=int, default=10, help="终端展示条数")
    p.add_argument("--skip-export", action="store_true",
                   help="跳过 infer 导出，直接复用 infer-root 已有窗口")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def run_infer(args, symbols):
    """逐股调用 dataset.py --mode infer（子进程，线程池并行）。"""
    infer_root = REPO / args.infer_root
    infer_root.mkdir(parents=True, exist_ok=True)
    results, failures = {}, {}
    if args.skip_export:
        return results, failures
    todo = symbols

    def one(symbol):
        cmd = [
            sys.executable, "dataset.py", "--mode", "infer",
            "--db", args.db,
            "--preprocessor", str(REPO / args.data_root / symbol / "preprocessing.json"),
            "--out-dir", str(infer_root),
        ]
        proc = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
        return symbol, proc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, s): s for s in sorted(todo)}
        done = 0
        for future in as_completed(futures):
            symbol, proc = future.result()
            done += 1
            if proc.returncode == 0:
                results[symbol] = "ok"
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                failures[symbol] = tail[-1][:160] if tail else "unknown error"
            if done % 25 == 0 or done == len(todo):
                print(f"infer 导出进度 {done}/{len(todo)}", flush=True)
    return results, failures


@torch.no_grad()
def score(args, saved, device):
    """加载全部 latest.npz，批量前向，返回打分明细表。"""
    infer_root = REPO / args.infer_root
    schema = saved["schema"]
    config = saved["model_config"]

    # 掩码协议跟随 checkpoint：训练时启用 ex_mask 的模型，打分时必须
    # 同样屏蔽窗口内整段无观测的外生值 token（判定只看 __missing 标记）。
    use_ex_mask = bool(config.get("ex_mask", False))
    ex_mask_layout = exogenous_mask_layout(schema)

    rows, arrays, masks, symbols = [], [], [], []
    for symbol, prep in saved["preprocessing"].items():
        path = infer_root / symbol / "latest.npz"
        if not path.exists():
            rows.append({"symbol": symbol, "note": "no_infer_output"})
            continue
        with np.load(path, allow_pickle=True) as latest:
            x = align_features(latest["X"], prep["feature_columns"], schema)
            input_date = str(latest["input_date"][0])
            target_date = str(latest["target_date"][0])
        if x.shape != (1, config["seq_len"], config["enc_in"]):
            rows.append({"symbol": symbol, "note": f"bad_shape_{x.shape}"})
            continue
        if not np.isfinite(x).all():
            rows.append({"symbol": symbol, "note": "non_finite"})
            continue
        arrays.append(x[0])
        if use_ex_mask:
            masks.append(compute_ex_mask(x, ex_mask_layout)[0])
        symbols.append(symbol)
        rows.append({
            "symbol": symbol, "note": "ok",
            "input_date": input_date, "target_date": target_date,
        })
    if not arrays:
        raise RuntimeError("没有可打分的推理窗口")

    model = build_model(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    preds = []
    for i in range(0, len(arrays), args.batch_size):
        batch = torch.from_numpy(np.stack(arrays[i:i + args.batch_size])).to(device)
        ex_mask = None
        if use_ex_mask:
            ex_mask = torch.from_numpy(
                np.stack(masks[i:i + args.batch_size])
            ).to(device)
        out = model(
            batch, None, None, None, ex_mask=ex_mask
        ).reshape(-1).cpu().numpy()
        preds.append(out)

    frame = pd.DataFrame(rows).set_index("symbol")
    for symbol, z in zip(symbols, np.concatenate(preds)):
        stats = saved["preprocessing"][symbol]["statistics"][TARGET]
        frame.loc[symbol, "pred_return_pct"] = (
            z * stats["scale"] + stats["mean"]
        ) * 100.0
    return frame.reset_index()


def main():
    args = parse_args()
    if min(args.workers, args.batch_size, args.top) < 1:
        raise ValueError("参数不合法")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["format"] != "global_timexer_typed_v1":
        raise ValueError("不支持的 checkpoint 格式")
    symbols = sorted(saved["preprocessing"])
    print(f"checkpoint 股票池：{len(symbols)} 只，device={device}")

    if args.skip_export:
        print("跳过 infer 导出（--skip-export）")
        failures = {}
    else:
        results, failures = run_infer(args, symbols)
        print(f"infer 完成：成功 {len(results)}，失败 {len(failures)}")
        for symbol, err in list(failures.items())[:5]:
            print(f"  失败示例 {symbol}: {err}")

    frame = score(args, saved, device)

    universe = pd.read_csv(
        REPO / args.data_root / "universe.csv",
        dtype={"symbol": str}, encoding="utf-8-sig",
    )[["symbol", "name", "industry_l2"]]
    frame = frame.merge(universe, on="symbol", how="left")

    ok = frame[frame.note == "ok"].copy()
    stale = ok[ok.input_date != ok.input_date.max()]
    valid = ok[ok.input_date == ok.input_date.max()].copy()

    input_date = ok.input_date.max()
    target_dates = valid.target_date.unique()
    target_date = sorted(target_dates)[-1]
    if len(target_dates) > 1:
        print(f"[WARN] 目标日不一致 {list(target_dates)}，以 {target_date} 为准")

    # 混合规则：排名前 q% 且预测 > 下限
    valid = valid.sort_values("pred_return_pct", ascending=False).reset_index(drop=True)
    n = len(valid)
    k = int(np.ceil(n * args.quantile_pct / 100.0))
    valid["rank"] = np.arange(1, n + 1)
    valid["quantile_pass"] = valid["rank"] <= k
    valid["floor_pass"] = valid.pred_return_pct > args.floor_pct
    valid["signal"] = valid.quantile_pass & valid.floor_pass
    valid.loc[valid.signal, "note"] = "signal"

    print(f"\n== 数据新鲜度 ==")
    print(f"输入行情日（t）：{input_date}  预测目标日（t+1）：{target_date}")
    stale_days = (
        pd.Timestamp.today().normalize() - pd.Timestamp(input_date)
    ).days
    if stale_days > 3:
        print(f"[WARN] 数据已落后约 {stale_days} 天，请先更新 AutoTS 下载库再跑！")
    print(f"打分 {len(ok)}/{len(symbols)}，停牌/滞后剔除 {len(stale)}，"
          f"infer 失败 {len(failures)}")
    if len(valid):
        print(f"预测分布：中位 {valid.pred_return_pct.median():.3f}%，"
              f"最大 {valid.pred_return_pct.max():.3f}%，"
              f"最小 {valid.pred_return_pct.min():.3f}%")

    signals = valid[valid.signal]
    print(f"\n== 混合规则信号（前 {args.quantile_pct:g}% 且 预测>{args.floor_pct:g}%）："
          f"{len(signals)} 只 ==")
    if len(signals):
        show = signals.head(args.top)[[
            "rank", "symbol", "name", "industry_l2", "pred_return_pct",
        ]].copy()
        show.pred_return_pct = show.pred_return_pct.round(3)
        show.columns = ["排名", "代码", "名称", "行业", "预测涨幅%"]
        print(show.to_string(index=False))
        if len(signals) > args.top:
            print(f"... 其余 {len(signals) - args.top} 只见 CSV")
    else:
        print("  今日无股票同时满足两条规则（信心不足，建议空仓观望）")

    top_view = valid.head(args.top)[[
        "rank", "symbol", "name", "industry_l2", "pred_return_pct", "signal",
    ]].copy()
    top_view.pred_return_pct = top_view.pred_return_pct.round(3)
    print(f"\n== 全池预测 Top {min(args.top, n)}（signal=是否通过混合规则）==")
    print(top_view.to_string(index=False))

    out_dir = REPO / "predictions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"predict_{target_date}.csv"
    columns = ["rank", "symbol", "name", "industry_l2", "input_date",
               "target_date", "pred_return_pct", "quantile_pass",
               "floor_pass", "signal", "note"]
    full = valid.reindex(columns=columns)
    full.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全池明细已保存：{out_path}")


if __name__ == "__main__":
    main()
