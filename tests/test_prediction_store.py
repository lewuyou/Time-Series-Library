import sqlite3

import pandas as pd
import pytest

from stock.production import prediction_store


def _frame(target_date="2026-10-15"):
    return pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "target_date": [target_date] * 3,
        "as_of_date": ["2026-10-14"] * 3,
        "model_id": ["m1"] * 3,
        "dataset_id": ["d1"] * 3,
        "y_pred": [0.01, float("nan"), float("nan")],
        "pred_return_pct": [1.0, float("nan"), float("nan")],
        "status": ["ok", "stale", "no_infer_output"],
        "rank": [1, None, None],
        "quantile_pass": [True, False, False],
        "floor_pass": [True, False, False],
        "signal": [True, False, False],
        "input_date": ["2026-10-14", "2026-10-13", None],
        "detail": [None, None, "missing latest.npz"],
    })


def _register(con, run_id, model_id="m1"):
    prediction_store.register_run(
        con, run_id=run_id, as_of_date="2026-10-14",
        target_date="2026-10-15", model_id=model_id, dataset_id="d1",
        checkpoint="/tmp/ckpt.pth", checkpoint_sha256="a" * 64,
        feature_schema_hash="b" * 64, quantile_pct=5, floor_pct=0.5,
        publish_status="published",
    )


def test_two_runs_same_target_date_are_both_readable(tmp_path):
    con = prediction_store.connect(tmp_path / "pred.db")
    try:
        _register(con, "run-a")
        prediction_store.insert_predictions(con, "run-a", _frame())
        _register(con, "run-b", model_id="m2")
        prediction_store.insert_predictions(con, "run-b", _frame())
        assert prediction_store.run_ids_for_target(con, "2026-10-15") == [
            "run-a", "run-b",
        ]
        a = prediction_store.read_predictions(con, "run-a")
        b = prediction_store.read_predictions(con, "run-b")
        assert len(a) == len(b) == 3
        assert set(a.status) == {"ok", "stale", "no_infer_output"}
        assert prediction_store.read_run(con, "run-b")["model_id"] == "m2"
    finally:
        con.close()


def test_insert_is_frozen_and_duplicates_raise(tmp_path):
    con = prediction_store.connect(tmp_path / "pred.db")
    try:
        _register(con, "run-a")
        prediction_store.insert_predictions(con, "run-a", _frame())
        with pytest.raises(prediction_store.FrozenPredictionError):
            prediction_store.insert_predictions(con, "run-a", _frame())
        with pytest.raises(sqlite3.IntegrityError, match="已冻结"):
            con.execute(
                "UPDATE predictions SET y_pred = 9.9 WHERE run_id = 'run-a'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="已冻结"):
            con.execute("DELETE FROM predictions WHERE run_id = 'run-a'")
    finally:
        con.close()


def test_export_view_marks_database_as_authoritative(tmp_path):
    con = prediction_store.connect(tmp_path / "pred.db")
    try:
        _register(con, "run-a")
        prediction_store.insert_predictions(con, "run-a", _frame())
        path = prediction_store.export_view(
            con, "run-a", tmp_path / "view.csv"
        )
        text = path.read_text(encoding="utf-8-sig")
        assert text.splitlines()[0].startswith("# 视图，以库为准")
        assert "run-a" in text.splitlines()[0]
        assert "no_infer_output" in text
    finally:
        con.close()
