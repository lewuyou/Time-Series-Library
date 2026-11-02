import pandas as pd

from stock.production import prediction_store
from stock.production.outcome_evaluator import (
    connect_outcomes,
    evaluate_run,
    fill_outcomes,
)


def _db(tmp_path):
    con = connect_outcomes(tmp_path / "pred.db")
    prediction_store.register_run(
        con, run_id="run-a", as_of_date="2026-10-13",
        target_date="2026-10-14", model_id="m", dataset_id="d",
        checkpoint="/tmp/c.pth", checkpoint_sha256="a" * 64,
        quantile_pct=5, floor_pct=0.5, publish_status="published",
    )
    frame = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "target_date": ["2026-10-14"] * 3,
        "as_of_date": ["2026-10-13"] * 3,
        "model_id": ["m"] * 3,
        "dataset_id": ["d"] * 3,
        "y_pred": [0.02, 0.015, -0.01],
        "pred_return_pct": [2.0, 1.5, -1.0],
        "status": ["ok"] * 3,
        "rank": [1, 2, 3],
        "quantile_pass": [True, True, False],
        "floor_pass": [True, True, False],
        "signal": [True, True, False],
        "input_date": ["2026-10-13"] * 3,
    })
    prediction_store.insert_predictions(con, "run-a", frame)
    return con


def test_fill_and_evaluate(tmp_path):
    con = _db(tmp_path)
    labels = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "target_date": ["2026-10-14"] * 3,
        "y_true": [0.01, -0.01, 0.005],
    })
    try:
        assert fill_outcomes(con, "run-a", labels) == 3
        summary = evaluate_run(con, "run-a")
        assert summary["coverage"] == 1.0
        assert summary["signals"] == 2
        assert summary["signal_hit_rate"] == 0.5
        assert summary["base_up_rate"] == 2 / 3
        assert summary["all_mean_return"] == (0.01 - 0.01 + 0.005) / 3
    finally:
        con.close()


def test_outcome_label_can_be_revised(tmp_path):
    con = _db(tmp_path)
    labels = pd.DataFrame({
        "symbol": ["000001"], "target_date": ["2026-10-14"],
        "y_true": [0.01],
    })
    try:
        fill_outcomes(con, "run-a", labels)
        labels.loc[0, "y_true"] = -0.02
        fill_outcomes(con, "run-a", labels)
        frame = con.execute(
            "SELECT realized_return FROM prediction_outcomes "
            "WHERE run_id='run-a' AND symbol='000001'"
        ).fetchall()
        assert len(frame) == 1 and frame[0][0] == -0.02
    finally:
        con.close()
