# -*- coding: utf-8 -*-
"""目标定义回归：公式与旧内联实现逐值一致，登记字段完整。"""

import numpy as np
import pandas as pd

from stock.data.schema import TARGET
from stock.data.targets import (
    TARGET_SPEC,
    label_valid,
    target_series,
    target_spec,
)


def test_target_series_matches_legacy_simple():
    rng = np.random.default_rng(20261102)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    )
    close.iloc[[3, 17, 99]] = np.nan
    expected = close / close.shift(1) - 1.0
    actual = target_series(close, "simple")
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_target_series_log_matches_legacy():
    close = pd.Series([10.0, 10.5, 9.8, np.nan, 11.0, 0.0])
    expected = np.log(close / close.shift(1))
    actual = target_series(close, "log")
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_target_series_rejects_unknown_return_type():
    try:
        target_series(pd.Series([1.0, 2.0]), "cumulative")
    except ValueError as exc:
        assert "return_type" in str(exc)
    else:
        raise AssertionError("未知 return_type 必须报错")


def test_label_valid_requires_finite_and_valid_window():
    assert label_valid(True, 0.01)
    assert not label_valid(True, np.nan)
    assert not label_valid(True, np.inf)
    assert not label_valid(False, 0.01)
    assert label_valid(np.bool_(True), np.float64(-0.03))
    assert not label_valid(np.bool_(False), np.float64(-0.03))


def test_target_spec_registry_fields():
    for key in (
        "name", "horizon", "definition", "unit", "price_basis",
        "target_session", "suspension", "decision_time",
    ):
        assert key in TARGET_SPEC
    assert TARGET_SPEC["name"] == TARGET
    assert TARGET_SPEC["horizon"] == 1
    assert TARGET_SPEC["unit"] == "fraction"
    assert TARGET_SPEC["price_basis"] == "adjusted_close"
    assert target_spec("log")["return_type"] == "log"
    assert target_spec("log")["name"] == TARGET
