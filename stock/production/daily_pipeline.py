# -*- coding: utf-8 -*-
"""每日选股预测：共享模型全池打分 + 混合规则选股（生产化第 4 步）。

自 predict_daily.py 原样迁入，结构性调整：
    1. 模型构造/前向与 checkpoint 格式校验改为经适配器注册表
       （stock.adapters），不再写死 TimeXer；
    2. 打分逻辑迁至 stock.inference.predictor.score（与单股 predict
       共用同一套加载/掩码协议）；
    3. 混合规则改用 stock.selection.rules.hybrid_signal（与
       analyze_daily_hit / walk-forward 共享同一实现）；
    4. 结果落 SQLite（stock.production.prediction_store）为唯一事实源，
       同日重跑新建 run_id 而非覆盖；CSV 只是导出视图；
    5. 发布前经 stock.production.readiness 判定，数据滞后/覆盖不足/
       坏 checkpoint 等直接阻断（exit 1），但仍保留本次 run 的事实记录。

流程（每个交易日收盘后运行，运行前请先更新 AutoTS 下载库）：
    1. 对 checkpoint 登记的每只股票调用 dataset.py --mode infer，
       用冻结的训练期预处理参数从 duckdb 生成推理窗口 latest.npz；
    2. 批量过模型得到下一交易日简单收益率预测（还原到原始尺度）；
    3. 混合规则选股（与回测口径一致）：
           当日预测值排名前 q%（默认 5%） 且 预测涨幅 > 下限（默认 0.5%）；
    4. 落库 + 就绪检查 + 导出 CSV 视图。

用法：
    # 完整流程（infer 导出 + 打分）
    python -m stock.production.daily_pipeline \
        --checkpoint checkpoints/global_final_001/checkpoint.pth \
        --data-root dataset/global_002 \
        --db ../AutoTS/download/autots.duckdb \
        --infer-root dataset/global_infer_001 \
        --workers 4 --top 10

    # 指定 as-of（重放历史某日；数据集必须已含该日之后的下一交易日）
    python -m stock.production.daily_pipeline ... --as-of 2026-10-30

    # 只重打分（复用已有 infer 导出，跳过步骤 1）
    python -m stock.production.daily_pipeline --checkpoint ... --data-root ... --skip-export

输出：
    终端：数据新鲜度 / 打分统计 / Top 榜（标注是否通过混合规则）
    predictions/predictions.db              唯一事实源（每次运行一个 run_id）
    predictions/predict_<target_date>.csv   导出视图（首行标注"以库为准"）

注意：
    - 预测的是 t+1 收盘相对 t 收盘的后复权简单收益率；信号在 t+1 早
      08:30 才能形成，无法按 t 收盘价成交，实际可交易价格见开盘。
    - 停牌/数据滞后的股票（input_date 落后于全池最新日）状态记为
      stale，不参与信号排名，但完整落库。
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stock.adapters import adapter_for_checkpoint
from stock.data.schema import feature_schema_doc
from stock.inference.contracts import (
    PredictionStatus,
    normalize_prediction_frame,
)
from stock.inference.predictor import score
from stock.production import prediction_store, readiness
from stock.selection.rules import hybrid_signal
from stock.utils import sha256_file

# repo/stock/production/daily_pipeline.py -> 仓库根
REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True,
                   help="训练期导出目录（读 preprocessing.json / universe.csv）")
    p.add_argument("--db", default="../AutoTS/download/autots.duckdb")
    p.add_argument("--infer-root", default="dataset/global_infer_001")
    p.add_argument("--run-db", default="predictions/predictions.db",
                   help="预测库（唯一事实源）；相对路径按仓库根解析")
    p.add_argument("--as-of",
                   help="推理基准日（t）；缺省取库内最新行情日")
    p.add_argument("--model-id",
                   help="覆盖模型标识；缺省取 checkpoint 记录或文件哈希")
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
    p.add_argument("--max-lag-days", type=int, default=5,
                   help="发布门槛：as_of 距今天数上限")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="发布门槛：可打分股票占 checkpoint 股票池比例")
    p.add_argument("--max-failure-ratio", type=float, default=0.2,
                   help="发布门槛：infer/打分失败占比上限")
    return p.parse_args()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


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
        if args.as_of:
            cmd.extend(["--as-of", args.as_of])
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


def build_records(args, frame, failures, symbols, model_id, dataset_id):
    """把打分明细整理成统一预测契约（失败/stale 全部保留）。"""
    frame = frame.copy()
    frame["model_id"] = model_id
    frame["dataset_id"] = dataset_id
    if "detail" not in frame:
        frame["detail"] = None

    # infer 子进程失败但旧窗口仍在时，score 会把旧数据当成功；这里以
    # 导出结果为准强制标 error，避免把过期结果当本次预测。
    if failures:
        bad = frame["symbol"].isin(failures)
        frame.loc[bad, "status"] = PredictionStatus.ERROR.value
        frame.loc[bad, "detail"] = frame.loc[bad, "symbol"].map(failures)
        frame.loc[bad, ["y_pred", "pred_return_pct"]] = np.nan

    ok_mask = frame["status"] == PredictionStatus.OK.value
    input_dates = frame.loc[ok_mask, "input_date"].dropna()
    max_input = input_dates.max() if len(input_dates) else None
    stale_mask = ok_mask & (frame["input_date"] != max_input)
    frame.loc[stale_mask, "status"] = PredictionStatus.STALE.value

    as_of = args.as_of or max_input
    if as_of is None:
        return frame, None, None
    frame["as_of_date"] = as_of
    frame.loc[ok_mask, "as_of_date"] = frame.loc[ok_mask, "input_date"].fillna(as_of)
    if max_input is not None:
        frame.loc[stale_mask, "as_of_date"] = frame.loc[stale_mask, "input_date"]
    frame["as_of_date"] = frame["as_of_date"].astype(str)
    return frame, str(pd.Timestamp(as_of).date()), max_input


def apply_selection(frame, q_pct, floor_pct):
    """对可用记录跑混合规则，并把 rank/signal 写回整张表。"""
    valid = frame[frame["status"] == PredictionStatus.OK.value].copy()
    valid = valid.dropna(subset=["pred_return_pct"])
    valid = valid.sort_values(
        ["pred_return_pct", "symbol"], ascending=[False, True]
    )
    rule = hybrid_signal(
        valid["pred_return_pct"], q_pct,
        floor_pct=floor_pct, pred_unit="pct",
        symbol=valid["symbol"], eligible=valid["symbol"].tolist(),
    )
    for column in ("rank", "quantile_pass", "floor_pass", "signal"):
        frame.loc[rule.index, column] = rule[column].to_numpy()
        valid[column] = rule[column].to_numpy()
    return valid, rule


def main():
    args = parse_args()
    if min(args.workers, args.batch_size, args.top) < 1:
        raise ValueError("参数不合法")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint_sha = sha256_file(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 格式校验：未知 checkpoint 格式在注册表反查时直接报错（与原
    # "global_timexer_typed_v1" 白名单同等的严格性，但不再写死模型）。
    adapter_for_checkpoint(saved.get("format"))
    symbols = sorted(saved["preprocessing"])
    print(f"checkpoint 股票池：{len(symbols)} 只，device={device}")

    model_id = (
        args.model_id or saved.get("model_id") or checkpoint_sha[:16]
    )
    dataset_id = saved.get("dataset_id")
    if not dataset_id:
        config_path = REPO / args.data_root / "export_config.json"
        dataset_id = (
            sha256_file(config_path) if config_path.exists() else "unknown"
        )
    schema_doc = saved.get("feature_schema") or feature_schema_doc(saved["schema"])
    schema_hash = schema_doc.get("hash")

    if args.skip_export:
        print("跳过 infer 导出（--skip-export）")
        failures = {}
    else:
        results, failures = run_infer(args, symbols)
        print(f"infer 完成：成功 {len(results)}，失败 {len(failures)}")
        for symbol, err in list(failures.items())[:5]:
            print(f"  失败示例 {symbol}: {err}")

    frame = score(args, saved, device, REPO / args.infer_root)

    universe = pd.read_csv(
        REPO / args.data_root / "universe.csv",
        dtype={"symbol": str}, encoding="utf-8-sig",
    )[["symbol", "name", "industry_l2"]]
    frame = frame.merge(universe, on="symbol", how="left")

    frame, as_of, max_input = build_records(
        args, frame, failures, symbols, model_id, dataset_id
    )
    if as_of is None:
        raise RuntimeError("没有任何成功打分的推理窗口，无法生成预测记录")

    ok = frame[frame["status"] == PredictionStatus.OK.value]
    stale = frame[frame["status"] == PredictionStatus.STALE.value]
    errored = frame[~frame["status"].isin([
        PredictionStatus.OK.value, PredictionStatus.STALE.value,
    ])]
    usable = frame[frame["status"].isin([
        PredictionStatus.OK.value, PredictionStatus.STALE.value,
    ])]
    if usable.empty:
        raise RuntimeError("没有任何可用预测记录（全部失败/无推理输出）")

    target_dates = sorted(set(usable["target_date"].astype(str)))
    target_date = target_dates[-1]
    if len(target_dates) > 1:
        print(f"[WARN] 目标日不一致 {target_dates}，以 {target_date} 为准")

    valid, rule = apply_selection(frame, args.quantile_pct, args.floor_pct)
    n = len(valid)

    print(f"\n== 数据新鲜度 ==")
    print(f"输入行情日（t）：{as_of}  预测目标日（t+1）：{target_date}")
    stale_days = (
        pd.Timestamp.now().normalize() - pd.Timestamp(as_of)
    ).days
    if stale_days > args.max_lag_days:
        print(f"[WARN] 数据已落后约 {stale_days} 天，发布将被阻断")
    print(f"打分 {len(ok)}/{len(symbols)}，stale {len(stale)}，"
          f"失败/无窗口 {len(errored)}")
    if n:
        print(f"预测分布：中位 {valid.pred_return_pct.median():.3f}%，"
              f"最大 {valid.pred_return_pct.max():.3f}%，"
              f"最小 {valid.pred_return_pct.min():.3f}%")

    signals = frame[frame["signal"] == True]  # noqa: E712  布尔列可能含 NA
    print(f"\n== 混合规则信号（前 {args.quantile_pct:g}% 且 预测>{args.floor_pct:g}%）："
          f"{len(signals)} 只 ==")
    if len(signals):
        show = signals.sort_values("rank").head(args.top)[[
            "rank", "symbol", "name", "industry_l2", "pred_return_pct",
        ]].copy()
        show.pred_return_pct = show.pred_return_pct.round(3)
        show.columns = ["排名", "代码", "名称", "行业", "预测涨幅%"]
        print(show.to_string(index=False))
        if len(signals) > args.top:
            print(f"... 其余 {len(signals) - args.top} 只见 CSV")
    else:
        print("  今日无股票同时满足两条规则（信心不足，建议空仓观望）")

    if n:
        top_view = valid.head(args.top)[[
            "rank", "symbol", "name", "industry_l2", "pred_return_pct", "signal",
        ]].copy()
        top_view.pred_return_pct = top_view.pred_return_pct.round(3)
        print(f"\n== 全池预测 Top {min(args.top, n)}（signal=是否通过混合规则）==")
        print(top_view.to_string(index=False))

    # 统一契约：缺字段直接报错，避免落库/导出各自解释。
    records = normalize_prediction_frame(frame)

    report = readiness.assess_publish(
        records, expected_symbols=symbols, as_of_date=as_of,
        target_date=target_date, max_lag_days=args.max_lag_days,
        min_coverage=args.min_coverage,
        max_failure_ratio=args.max_failure_ratio,
    )
    if args.as_of and max_input is not None and pd.Timestamp(max_input) < pd.Timestamp(args.as_of):
        report["blockers"].append(
            f"请求 as_of={args.as_of} 但行情只到 {max_input}"
        )
        report["ready"] = False

    publish_status = "published" if report["ready"] else "blocked"
    db_path = resolve_path(args.run_db)
    con = prediction_store.connect(db_path)
    try:
        run_id = prediction_store.make_run_id(
            con, target_date, checkpoint_sha
        )
        prediction_store.register_run(
            con, run_id=run_id, as_of_date=as_of, target_date=target_date,
            model_id=model_id, dataset_id=dataset_id,
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            feature_schema_hash=schema_hash,
            quantile_pct=args.quantile_pct, floor_pct=args.floor_pct,
            publish_status=publish_status,
            blockers=report["blockers"],
        )
        prediction_store.insert_predictions(con, run_id, records)
        print(f"\n已落库 run_id={run_id}（{db_path}；状态 {publish_status}）")

        if not report["ready"]:
            for blocker in report["blockers"]:
                print(f"[BLOCK] {blocker}", file=sys.stderr)
            print("[BLOCK] 发布被阻断：本次 run 已留档，但不导出 CSV 视图",
                  file=sys.stderr)
            raise SystemExit(1)

        out_dir = REPO / "predictions"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"predict_{target_date}.csv"
        prediction_store.export_view(con, run_id, out_path, universe=universe)
        print(f"CSV 导出视图（以库为准）：{out_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
