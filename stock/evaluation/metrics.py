"""预测指标与测试报告（全仓唯一实现）。

自 train_global_timexer.py 原样迁出（metrics / save_test_report），
walk-forward 与统一训练器共用，避免各处指标口径漂移。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stock.utils import write_json


def metrics(frame, column):
    y = frame.y_true.to_numpy()
    pred = frame[column].to_numpy()
    error = pred - y
    nonzero = y != 0
    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "direction_accuracy_nonzero": (
            float((np.sign(pred[nonzero]) == np.sign(y[nonzero])).mean())
            if nonzero.any() else None
        ),
    }


def save_test_report(frame, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictors = [
        "y_pred", "baseline_zero", "baseline_mean",
        "baseline_persistence",
    ]
    per_stock = []
    summary = {}

    for column in predictors:
        stock_rows = []
        for symbol, part in frame.groupby("symbol"):
            row = {"symbol": symbol, "predictor": column}
            row.update(metrics(part, column))
            per_stock.append(row)
            stock_rows.append(row)

        daily_ic = []
        for _, part in frame.groupby("target_date"):
            if (
                len(part) >= 10
                and part.y_true.nunique() > 1
                and part[column].nunique() > 1
            ):
                value = part.y_true.rank().corr(part[column].rank())
                if np.isfinite(value):
                    daily_ic.append(float(value))

        summary[column] = {
            "pooled": metrics(frame, column),
            "stock_equal_mae": float(np.mean([
                row["mae"] for row in stock_rows
            ])),
            "stock_equal_mse": float(np.mean([
                row["mse"] for row in stock_rows
            ])),
            "mean_daily_rank_ic": (
                float(np.mean(daily_ic)) if daily_ic else None
            ),
            "rank_ic_days": len(daily_ic),
        }

    # 口径声明（真缺口 #6 残留）：防止把重叠窗口数误读成独立样本量。
    summary["_semantics"] = {
        "n_semantics": "overlapping_windows",
        "n_note": (
            "pooled/stock_equal_* 的样本数 n 是重叠滚动窗口数"
            "（相邻窗口共享 seq_len-1 个样本），样本间不独立，"
            "不能当作独立样本量用于显著性检验"
        ),
        "daily_ic_semantics": (
            "mean_daily_rank_ic 按 target_date 逐日聚合"
            "（日度近似独立）；rank_ic_days 为参与聚合的交易日数"
        ),
    }

    frame.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(per_stock).to_csv(
        out / "test_metrics_by_stock.csv", index=False
    )
    write_json(out / "test_metrics.json", summary)
    return summary
