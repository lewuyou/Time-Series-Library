"""预测结果回填与评价：标签可获得后再算真实收益。

预测落库是"当时的事实"（不可变）；真实收益要等目标日收盘后才知道。
本模块把标签回填到独立的 `prediction_outcomes` 表（与冻结的
`predictions` 表解耦，允许标签修订），并按 run 汇总：

- 覆盖率：有标签的预测 / 全部预测；
- 信号股（signal=1）的命中率与平均真实收益；
- 全池上涨占比（基准）与信号超额；
- 非零方向准确率。

标签来源：CSV（symbol,target_date,y_true），y_true 为小数收益率。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock.production.prediction_store import connect, read_predictions
from stock.utils import now_text, write_json

OUTCOME_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    run_id           TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    target_date      TEXT NOT NULL,
    realized_return  REAL,
    y_pred           REAL,
    signal           INTEGER,
    label_source     TEXT,
    evaluated_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, target_date)
);
"""


def connect_outcomes(db_path):
    con = connect(db_path)
    con.executescript(OUTCOME_SCHEMA)
    con.commit()
    return con


def load_labels(path):
    frame = pd.read_csv(path, dtype={"symbol": str})
    missing = {"symbol", "target_date", "y_true"} - set(frame.columns)
    if missing:
        raise ValueError(f"标签文件缺少列：{sorted(missing)}")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["target_date"] = frame["target_date"].astype(str)
    frame = frame.dropna(subset=["y_true"])
    return frame[["symbol", "target_date", "y_true"]].drop_duplicates(
        ["symbol", "target_date"], keep="last"
    )


def fill_outcomes(con, run_id, labels, label_source="labels.csv"):
    """回填真实收益（同一 key 可修订标签，用 INSERT OR REPLACE）。"""
    predictions = read_predictions(con, run_id)
    merged = predictions.merge(
        labels, on=["symbol", "target_date"], how="inner", suffixes=("", "_label")
    )
    stamp = str(now_text())
    rows = [
        (
            run_id, row.symbol, row.target_date, float(row.y_true),
            None if pd.isna(row.y_pred) else float(row.y_pred),
            None if pd.isna(row.signal) else int(bool(row.signal)),
            label_source, stamp,
        )
        for row in merged.itertuples(index=False)
    ]
    with con:
        con.executemany(
            """INSERT OR REPLACE INTO prediction_outcomes (
                run_id, symbol, target_date, realized_return, y_pred, signal,
                label_source, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def evaluate_run(con, run_id):
    """按 run 汇总已回填标签的表现（未回填的预测不计入比率分母）。"""
    predictions = read_predictions(con, run_id)
    outcomes = pd.read_sql_query(
        "SELECT * FROM prediction_outcomes WHERE run_id = ?",
        con, params=(run_id,),
    )
    labeled = outcomes.dropna(subset=["realized_return"])
    signals = labeled[labeled["signal"] == 1]
    base_up = float((labeled["realized_return"] > 0).mean()) if len(labeled) else None
    signal_hit = (
        float((signals["realized_return"] > 0).mean()) if len(signals) else None
    )
    nonzero = labeled["realized_return"] != 0
    direction = (
        float(
            (np.sign(labeled.loc[nonzero, "y_pred"])
             == np.sign(labeled.loc[nonzero, "realized_return"])).mean()
        ) if nonzero.any() else None
    )
    return {
        "run_id": run_id,
        "predictions": int(len(predictions)),
        "labeled": int(len(labeled)),
        "coverage": (
            float(len(labeled) / len(predictions)) if len(predictions) else None
        ),
        "signals": int(len(signals)),
        "signal_hit_rate": signal_hit,
        "base_up_rate": base_up,
        "signal_excess_hit": (
            signal_hit - base_up
            if signal_hit is not None and base_up is not None else None
        ),
        "signal_mean_return": (
            float(signals["realized_return"].mean()) if len(signals) else None
        ),
        "all_mean_return": (
            float(labeled["realized_return"].mean()) if len(labeled) else None
        ),
        "direction_accuracy_nonzero": direction,
    }


def parse_args():
    p = argparse.ArgumentParser(description="回填预测真实收益并评价")
    p.add_argument("--run-db", default="predictions/predictions.db")
    p.add_argument("--run-id", required=True)
    p.add_argument("--labels", required=True,
                   help="CSV：symbol,target_date,y_true（小数收益率）")
    p.add_argument("--out", help="可选：评价报告 JSON 输出路径")
    return p.parse_args()


def main():
    args = parse_args()
    labels = load_labels(args.labels)
    con = connect_outcomes(args.run_db)
    try:
        filled = fill_outcomes(con, args.run_id, labels)
        summary = evaluate_run(con, args.run_id)
    finally:
        con.close()
    print(f"回填 {filled} 条（标签文件 {len(labels)} 条）")
    print(pd.Series(summary).to_string())
    if args.out:
        write_json(Path(args.out), summary)


if __name__ == "__main__":
    main()
