"""发布前就绪检查：不满足条件即阻断，而不是打印告警后照常发布。

历史 `daily_pipeline.py` 只对数据滞后 `print("[WARN] ...")`，覆盖不足
或 checkpoint 坏掉时仍会写 CSV，下游无法区分"今天没信号"和"今天
数据/模型不可用"。本模块把发布门槛收敛为可测试的判定：

- 数据新鲜度：as_of 距今天数 ≤ max_lag_days；
- 覆盖率：可打分股票数 / checkpoint 股票池 ≥ min_coverage；
- 失败率：infer/打分失败占比 ≤ max_failure_ratio；
- 目标日唯一：所有可用记录 target_date 一致；
- 特征契约：checkpoint 记录 feature_schema_hash；
- checkpoint 可加载：坏文件/缺字段直接列为阻断项。

`assess_publish` 只做判定，`enforce` 在阻断时抛 ReadinessError；
CLI 可直接对库内某次 run 复检（exit code 1 = 阻断）。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from stock.inference.contracts import PredictionStatus
from stock.production.prediction_store import (
    read_predictions,
    read_run,
)
from stock.utils import date_text


class ReadinessError(RuntimeError):
    """发布门槛未通过。"""


def checkpoint_meta(path):
    """读取 checkpoint 元信息；坏文件/缺关键字段直接报错。"""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in saved or "schema" not in saved:
        raise ValueError(f"checkpoint 缺少 state_dict/schema：{path}")
    preprocessing = saved.get("preprocessing")
    if not isinstance(preprocessing, dict) or not preprocessing:
        raise ValueError(f"checkpoint 缺少 preprocessing 登记：{path}")
    return {
        "model_id": saved.get("model_id"),
        "dataset_id": saved.get("dataset_id"),
        "feature_schema_hash": (saved.get("feature_schema") or {}).get("hash"),
        "format": saved.get("format"),
        "symbols": sorted(preprocessing),
    }


def assess_publish(frame, *, expected_symbols, as_of_date, target_date,
                   today=None, max_lag_days=5, min_coverage=0.8,
                   max_failure_ratio=0.2):
    """判定本次运行能否发布；返回结构化报告，不抛异常。"""
    expected = len(set(map(str, expected_symbols)))
    if expected == 0:
        raise ValueError("expected_symbols 为空，无法评估覆盖率")
    as_of = date_text(as_of_date)
    target = date_text(target_date)
    today_text = date_text(today if today is not None else pd.Timestamp.now())

    status = frame["status"].astype(str)
    ok = status == PredictionStatus.OK.value
    stale = status == PredictionStatus.STALE.value
    failed = ~(ok | stale)

    lag_days = (pd.Timestamp(today_text) - pd.Timestamp(as_of)).days
    coverage = float(ok.sum()) / expected
    failure_ratio = float(failed.sum()) / expected

    usable = frame[ok | stale]
    target_dates = sorted(set(usable["target_date"].astype(str)))
    target_consistent = target_dates == [target]

    blockers = []
    if lag_days > max_lag_days:
        blockers.append(
            f"数据滞后 {lag_days} 天 > 允许 {max_lag_days} 天（as_of={as_of}）"
        )
    if coverage < min_coverage:
        blockers.append(
            f"覆盖率 {coverage:.1%} < 要求 {min_coverage:.0%}"
            f"（可打分 {int(ok.sum())}/{expected}）"
        )
    if failure_ratio > max_failure_ratio:
        blockers.append(
            f"失败占比 {failure_ratio:.1%} > 允许 {max_failure_ratio:.0%}"
            f"（失败 {int(failed.sum())}/{expected}）"
        )
    if not target_consistent:
        blockers.append(
            f"目标日不一致：{target_dates}，期望 [{target}]"
        )
    if int(ok.sum()) == 0:
        blockers.append("没有可打分记录（全部为失败/停牌）")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "as_of_date": as_of,
        "target_date": target,
        "today": today_text,
        "counts": {
            "expected": expected,
            "ok": int(ok.sum()),
            "stale": int(stale.sum()),
            "failed": int(failed.sum()),
        },
        "checks": {
            "lag_days": int(lag_days),
            "max_lag_days": int(max_lag_days),
            "coverage": coverage,
            "min_coverage": float(min_coverage),
            "failure_ratio": failure_ratio,
            "max_failure_ratio": float(max_failure_ratio),
            "target_dates": target_dates,
            "target_date_consistent": bool(target_consistent),
        },
    }


def enforce(report):
    """阻断项非空时抛 ReadinessError（调用方据此停止发布）。"""
    if not report["ready"]:
        raise ReadinessError(
            "发布被阻断：" + "；".join(report["blockers"])
        )
    return report


def parse_args():
    p = argparse.ArgumentParser(description="对库内某次预测运行做发布就绪复检")
    p.add_argument("--run-db", default="predictions/predictions.db")
    p.add_argument("--run-id", required=True)
    p.add_argument("--checkpoint", help="可选：用 checkpoint 股票池复检覆盖率")
    p.add_argument("--today", help="覆盖'今天'（复检历史运行时用）")
    p.add_argument("--max-lag-days", type=int, default=5)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--max-failure-ratio", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    from stock.production.prediction_store import connect

    con = connect(args.run_db)
    try:
        run = read_run(con, args.run_id)
        frame = read_predictions(con, args.run_id)
    finally:
        con.close()

    blockers = []
    expected_symbols = None
    feature_schema_hash = run.get("feature_schema_hash")
    if args.checkpoint:
        try:
            meta = checkpoint_meta(args.checkpoint)
        except Exception as exc:  # 坏 checkpoint 本身就是阻断项
            blockers.append(f"checkpoint 不可用：{exc}")
            expected_symbols = []
        else:
            expected_symbols = meta["symbols"]
            if not feature_schema_hash:
                feature_schema_hash = meta["feature_schema_hash"]
    if not feature_schema_hash:
        blockers.append("缺少 feature_schema_hash（无法确认特征契约）")
    if expected_symbols is None:
        expected_symbols = sorted(set(frame["symbol"].astype(str)))
    if not expected_symbols:
        blockers.append("无法确定 checkpoint 股票池")

    if expected_symbols:
        report = assess_publish(
            frame, expected_symbols=expected_symbols,
            as_of_date=run["as_of_date"], target_date=run["target_date"],
            today=args.today, max_lag_days=args.max_lag_days,
            min_coverage=args.min_coverage,
            max_failure_ratio=args.max_failure_ratio,
        )
        blockers.extend(report["blockers"])
    else:
        report = {"ready": False, "counts": {}}

    report["ready"] = not blockers
    report["blockers"] = blockers
    print(pd.Series(report, dtype=object).to_string())
    if not report["ready"]:
        print("\n".join(f"[BLOCK] {b}" for b in blockers), file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
