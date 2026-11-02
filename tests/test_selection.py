# -*- coding: utf-8 -*-
"""选股规则契约：单位、并列、eligible 分母、空组合与错误用法。"""

import numpy as np
import pandas as pd

from stock.selection.rules import hybrid_signal


def _frame(n=100, seed=20261102):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(n)],
        "pred_decimal": rng.normal(0.002, 0.01, n),
    })


def test_pct_and_decimal_units_are_equivalent():
    frame = _frame()
    pct = hybrid_signal(
        frame.pred_decimal * 100, 5,
        floor_pct=0.5, pred_unit="pct",
        symbol=frame.symbol, eligible=frame.symbol.tolist(),
    )
    decimal = hybrid_signal(
        frame.pred_decimal, 5,
        floor_decimal=0.5 / 100.0, pred_unit="decimal",
        symbol=frame.symbol, eligible=frame.symbol.tolist(),
    )
    assert pct.signal.equals(decimal.signal)
    assert pct.floor_pass.equals(decimal.floor_pass)
    assert pct["rank"].equals(decimal["rank"])
    # 0.5% 阈值：小数口径必须换算成 0.005，不是 0.5
    assert decimal.attrs["floor_threshold"] == 0.005


def test_legacy_positional_floor_is_raw_same_unit():
    frame = _frame()
    legacy = hybrid_signal(frame.pred_decimal, 5, 0.005)
    explicit = hybrid_signal(
        frame.pred_decimal, 5,
        floor_decimal=0.005, pred_unit="decimal",
        symbol=frame.symbol, eligible=frame.symbol.tolist(),
    )
    assert legacy.signal.equals(explicit.signal)


def test_tie_break_is_pred_desc_then_symbol_asc():
    pred = pd.Series([0.02, 0.02, 0.02, 0.01])
    symbol = pd.Series(["300001", "000001", "200001", "000002"])
    result = hybrid_signal(
        pred, 50, floor_decimal=0.0, pred_unit="decimal",
        symbol=symbol,
    )
    # 前 50% = 2 个：并列 0.02 中按 symbol 升序取 000001/200001
    assert result.loc[result.signal].index.tolist() == [1, 2]
    assert result.loc[0, "rank"] == 3
    assert result.loc[3, "rank"] == 4


def test_eligible_universe_controls_denominator():
    frame = _frame(10)
    eligible = frame.symbol.iloc[:2].tolist()
    result = hybrid_signal(
        frame.pred_decimal, 50,
        floor_decimal=0.0, pred_unit="decimal",
        symbol=frame.symbol, eligible=eligible,
    )
    # 分母只看 2 只 -> k=1；非 eligible 不参与排名
    assert int(result.signal.sum()) == 1
    assert result["rank"].isna().sum() == 8
    assert result.loc[result.signal].iloc[0].eligible


def test_empty_prediction_returns_empty_result():
    result = hybrid_signal(
        pd.Series(dtype=float), 5,
        floor_decimal=0.005, pred_unit="decimal",
        symbol=pd.Series(dtype=str), eligible=[],
    )
    assert result.empty
    assert set(result.columns) == {
        "rank", "quantile_pass", "floor_pass", "signal", "eligible",
    }


def test_floor_arguments_are_exclusive_and_typed():
    pred = pd.Series([0.01, 0.02])
    for kwargs in (
        {},
        {"floor_decimal": 0.005, "floor_pct": 0.5},
        {"floor": 1.0, "floor_decimal": 0.005},
    ):
        try:
            hybrid_signal(pred, 5, **kwargs)
        except TypeError as exc:
            assert "floor" in str(exc)
        else:
            raise AssertionError("floor 参数组合必须报错")

    try:
        hybrid_signal(pred, 5, floor_decimal=0.005, pred_unit=None)
    except ValueError as exc:
        assert "pred_unit" in str(exc)
    else:
        raise AssertionError("缺 pred_unit 必须报错")

    try:
        hybrid_signal(
            pred, 5, floor_decimal=0.005, pred_unit="decimal",
            eligible=["000001"],
        )
    except ValueError as exc:
        assert "symbol" in str(exc)
    else:
        raise AssertionError("eligible 无 symbol 必须报错")
