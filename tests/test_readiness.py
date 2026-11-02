import pandas as pd
import pytest
import torch

from stock.production.readiness import (
    ReadinessError,
    assess_publish,
    checkpoint_meta,
    enforce,
)


def _frame(statuses, target_dates=None):
    n = len(statuses)
    return pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(n)],
        "status": statuses,
        "as_of_date": ["2026-10-14"] * n,
        "target_date": target_dates or ["2026-10-15"] * n,
    })


def test_ready_when_fresh_and_covered():
    frame = _frame(["ok"] * 9 + ["stale"])
    report = assess_publish(
        frame, expected_symbols=frame.symbol.tolist(),
        as_of_date="2026-10-14", target_date="2026-10-15",
        today="2026-10-15",
    )
    assert report["ready"], report["blockers"]
    assert report["counts"] == {"expected": 10, "ok": 9, "stale": 1, "failed": 0}
    enforce(report)


def test_stale_data_is_blocked():
    report = assess_publish(
        _frame(["ok"] * 10), expected_symbols=[f"{i:06d}" for i in range(10)],
        as_of_date="2026-10-14", target_date="2026-10-15",
        today="2026-11-02", max_lag_days=5,
    )
    assert not report["ready"]
    assert any("数据滞后" in b for b in report["blockers"])
    with pytest.raises(ReadinessError, match="阻断"):
        enforce(report)


def test_low_coverage_and_failures_are_blocked():
    frame = _frame(["ok"] + ["no_infer_output"] * 9)
    report = assess_publish(
        frame, expected_symbols=frame.symbol.tolist(),
        as_of_date="2026-10-14", target_date="2026-10-15",
        today="2026-10-15",
    )
    assert not report["ready"]
    text = "；".join(report["blockers"])
    assert "覆盖率" in text and "失败占比" in text


def test_inconsistent_target_date_is_blocked():
    frame = _frame(["ok"] * 10, ["2026-10-15"] * 5 + ["2026-10-16"] * 5)
    report = assess_publish(
        frame, expected_symbols=frame.symbol.tolist(),
        as_of_date="2026-10-14", target_date="2026-10-15",
        today="2026-10-15",
    )
    assert not report["ready"]
    assert any("目标日不一致" in b for b in report["blockers"])


def test_broken_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pth"
    path.write_text("not a checkpoint")
    with pytest.raises(Exception):
        checkpoint_meta(path)

    path2 = tmp_path / "empty.pth"
    torch.save({"format": "x"}, path2)
    with pytest.raises(ValueError, match="state_dict/schema"):
        checkpoint_meta(path2)
