# -*- coding: utf-8 -*-
"""分析最终模型 30 天实际验证期的"预测上涨 vs 实际上涨"命中率，挑出最好的 10 支股票。

输入：checkpoints/<run>/test_predictions.csv（train_global_timexer.py 输出）
输出：
    <run>/hit_rate_by_stock.csv   每股命中率明细
    <run>/top10_stocks.csv        命中率最高的 10 支股票
    终端摘要

口径：
    test 集即导出时保留的最后 30 个交易日（每日 1 个样本/股）。
    pred_up   : y_pred  > 0 的天数
    actual_up : y_true  > 0 的天数
    hit_days  : y_pred > 0 且 y_true > 0 的天数
    up_precision   = hit_days / pred_up    （预测上涨的命中率）
    up_recall      = hit_days / actual_up  （实际上涨被捕捉比例）
    dir_accuracy   = sign(y_pred)==sign(y_true) 占比
选股规则：pred_up >= MIN_PRED_UP（避免只预测少数几天上涨的样本偏差），
    按 up_precision 降序、并列时按 dir_accuracy 再按 pred_up。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="checkpoints/global_final_001")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-pred-up", type=int, default=15,
                   help="入选要求的最少预测上涨天数（30 天中至少一半）")
    return p.parse_args()


def main():
    args = parse_args()
    run = Path(args.run_dir)
    frame = pd.read_csv(run / "test_predictions.csv", dtype={"symbol": str})
    universe = pd.read_csv(
        "dataset/global_002/universe.csv",
        dtype={"symbol": str}, encoding="utf-8-sig",
    )[["symbol", "name", "industry_l2", "total_mv", "mv_rank"]]

    pred_up = frame.y_pred > 0
    actual_up = frame.y_true > 0
    hit = pred_up & actual_up
    sign_match = np.sign(frame.y_pred) == np.sign(frame.y_true)

    frame = frame.assign(pred_up=pred_up, actual_up=actual_up,
                         hit=hit, sign_match=sign_match)

    rows = []
    for symbol, part in frame.groupby("symbol"):
        pred_days = int(part.pred_up.sum())
        rows.append({
            "symbol": symbol,
            "test_days": len(part),
            "pred_up_days": pred_days,
            "actual_up_days": int(part.actual_up.sum()),
            "hit_days": int(part.hit.sum()),
            "up_precision": (part.hit.sum() / pred_days) if pred_days else np.nan,
            "up_recall": (
                part.hit.sum() / part.actual_up.sum()
                if part.actual_up.sum() else np.nan
            ),
            "dir_accuracy": part.sign_match.mean(),
            "avg_pred_return_pct": part.y_pred.mean() * 100,
            "avg_true_return_pct": part.y_true.mean() * 100,
            "cum_true_return_pct": ((1 + part.y_true).prod() - 1) * 100,
        })
    table = pd.DataFrame(rows).merge(universe, on="symbol", how="left")

    # 整体口径（全部股票合并）
    pooled_precision = frame.hit.sum() / frame.pred_up.sum()
    pooled_dir = frame.sign_match.mean()
    base_rate = frame.actual_up.mean()

    qualified = table[table.pred_up_days >= args.min_pred_up].copy()
    top = qualified.sort_values(
        ["up_precision", "dir_accuracy", "pred_up_days"],
        ascending=False,
    ).head(args.top)

    table.to_csv(run / "hit_rate_by_stock.csv", index=False)
    top.to_csv(run / "top10_stocks.csv", index=False)

    show = top[[
        "symbol", "name", "industry_l2", "pred_up_days", "actual_up_days",
        "hit_days", "up_precision", "dir_accuracy",
        "avg_pred_return_pct", "avg_true_return_pct", "cum_true_return_pct",
    ]].copy()
    for col in ("up_precision", "dir_accuracy"):
        show[col] = (show[col] * 100).round(1)
    for col in ("avg_pred_return_pct", "avg_true_return_pct",
                "cum_true_return_pct"):
        show[col] = show[col].round(2)

    print(f"整体：预测上涨天数={int(frame.pred_up.sum())}, "
          f"其中实际上涨={int(frame.hit.sum())}, "
          f"命中率={pooled_precision:.1%}（30天实际上涨占比 {base_rate:.1%}）；"
          f"方向一致率={pooled_dir:.1%}")
    print(f"入选门槛：预测上涨天数 >= {args.min_pred_up} 的股票 "
          f"{len(qualified)}/{len(table)} 只")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
