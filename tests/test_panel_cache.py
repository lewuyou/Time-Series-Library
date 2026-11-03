# -*- coding: utf-8 -*-
"""PanelWindows 矩阵缓存（LRU 上限与淘汰）行为测试。"""

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
for _p in (str(REPO), str(TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stock.data.panel import PanelWindows, load_catalog  # noqa: E402
from test_calendar_features import (  # noqa: E402
    ALL_CAL,
    TARGET,
    VALUE_COLS,
    _make_data_root,
)


def _windows(tmp_path, n_symbols):
    full = VALUE_COLS + ALL_CAL + [TARGET]
    root = _make_data_root(
        tmp_path, {chr(ord("A") + i): full for i in range(n_symbols)}
    )
    records, schema = load_catalog(root)
    return PanelWindows(records, schema, "train")


def test_matrix_cache_default_limit():
    # 对照实验结论固化为缺省上限（见 stock/data/panel.py 注释）
    assert PanelWindows._matrix_cache_limit == 512


def test_matrix_cache_evicts_least_recently_used(tmp_path):
    ds = _windows(tmp_path, 3)
    ds._matrix_cache_limit = 2  # 收紧上限以验证淘汰机制
    m0 = ds.matrix(0)
    ds.matrix(1)
    ds.matrix(2)
    assert len(ds.cache) <= 2
    assert 0 not in ds.cache  # 最久未用者被淘汰
    # 被淘汰后重新访问仍得到同形状、同数值的数据
    again = ds.matrix(0)
    assert again.shape == m0.shape
    np.testing.assert_array_equal(np.asarray(again), np.asarray(m0))
