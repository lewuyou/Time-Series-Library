# -*- coding: utf-8 -*-
"""样本切分回归：按 target_date 划分，标签无效仍保留行。"""

import numpy as np
import pandas as pd

import dataset as ds
from stock.data.schema import TARGET


def _features(days=12):
    index = pd.date_range("2024-01-02", periods=days, freq="B")
    rng = np.random.default_rng(20261102)
    return pd.DataFrame(
        {TARGET: rng.normal(0, 0.01, days)}, index=index,
    )


def test_split_is_by_target_date_not_input_date():
    features = _features(12)
    valid = pd.Series(True, index=features.index)
    samples, candidates, exclusions = ds.make_samples(
        features, valid, {"seq_len": 3},
        train_end=pd.Timestamp("2024-01-05"),
        val_end=pd.Timestamp("2024-01-09"),
    )
    assert candidates == 11
    assert exclusions == {
        "insufficient_history": 2,
        "invalid_input_window": 0,
        "invalid_target_label": 0,
    }
    assert len(samples) == 9
    for _, row in samples.iterrows():
        expected = (
            "train" if row.target_date <= pd.Timestamp("2024-01-05")
            else "val" if row.target_date <= pd.Timestamp("2024-01-09")
            else "test"
        )
        assert row.split == expected
        assert row.input_date < row.target_date
        assert row.target_pos == row.end_pos + 1
        assert row.end_pos - row.start_pos + 1 == 3


def test_invalid_input_window_is_dropped():
    features = _features(12)
    valid = pd.Series(True, index=features.index)
    valid.iloc[4] = False  # 落在某个 3 日窗口内
    samples, _, exclusions = ds.make_samples(
        features, valid, {"seq_len": 3},
        train_end=pd.Timestamp("2024-01-05"),
        val_end=pd.Timestamp("2024-01-09"),
    )
    assert exclusions["invalid_input_window"] >= 1
    assert not samples.empty


def test_invalid_target_label_kept_for_coverage():
    features = _features(12)
    valid = pd.Series(True, index=features.index)
    features.loc[features.index[-1], TARGET] = np.nan
    samples, _, exclusions = ds.make_samples(
        features, valid, {"seq_len": 3},
        train_end=pd.Timestamp("2024-01-05"),
        val_end=pd.Timestamp("2024-01-09"),
    )
    last = samples.iloc[-1]
    assert exclusions["invalid_target_label"] == 1
    assert not last.label_valid
    assert np.isnan(last.y_raw)
    # 无效标签只影响可评分，不代表"没有生成预测"
    assert last.input_date < last.target_date


def test_train_test_layout_has_no_validation():
    features = _features(10)
    valid = pd.Series(True, index=features.index)
    samples, _, _ = ds.make_samples(
        features, valid, {"seq_len": 2},
        train_end=pd.Timestamp("2024-01-08"),
        val_end=None,
        split_layout="train_test",
    )
    assert set(samples.split) == {"train", "test"}
    try:
        ds.make_samples(
            features, valid, {"seq_len": 2},
            train_end=pd.Timestamp("2024-01-08"),
            val_end=pd.Timestamp("2024-01-09"),
            split_layout="train_test",
        )
    except ValueError as exc:
        assert "val_end" in str(exc)
    else:
        raise AssertionError("train_test + val_end 必须报错")
