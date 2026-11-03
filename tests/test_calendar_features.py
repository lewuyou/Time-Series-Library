# -*- coding: utf-8 -*-
"""目标日日历特征 / calendar_mode 通道 / 预测头日历向量回归测试。

新约束版设计（放弃向后兼容，官方通道优先）：
- 日历列共 8 个：3 个官方时间标记列（target_mark_dow/dom/doy，
  日频归一化公式，按目标交易日取值）+ 5 个交易日历旗标列；
  原 target_dow_* one-hot 已删除（星期信息由 x_mark 与预测头承载）。
- calendar_mode ∈ {time_marks(默认), columns, none}：
  time_marks = 3 个标记列移入 x_mark（官方通道）；columns = 全部日历列
  留在输入列；none = 关闭全部日历信息。
- head_calendar_dim = 5（目标日星期 one-hot）默认开启，
  --no-head-calendar 关闭；checkpoint 格式 v1（关）/ v2（开）。
- 硬要求：models/TimeXer.py 开关全关时与 thuml 上游原版数值等价
  （参考实现 tests/reference_timexer_upstream.py，最大绝对差 <= 1e-6）。
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import reference_timexer_upstream as ref_upstream  # noqa: E402

from models.TimeXer import Model as TimeXer  # noqa: E402
from stock.adapters import get_adapter  # noqa: E402
from stock.data.features import target_calendar_features  # noqa: E402
from stock.data.panel import PanelWindows, load_catalog  # noqa: E402
from stock.data.preprocessing import fit_preprocessing  # noqa: E402
from stock.data.schema import (  # noqa: E402
    TARGET,
    TARGET_GAP_DAYS_LOG,
    TARGET_IS_GAP_GE4,
    TARGET_IS_MONTH_END_3,
    TARGET_IS_MONTH_START_3,
    TARGET_IS_PRE_LONG_HOLIDAY,
    TARGET_MARK_DOM,
    TARGET_MARK_DOY,
    TARGET_MARK_DOW,
    parse_calendar_features,
    target_calendar_columns,
)

ALL_CAL = target_calendar_columns()
MARK_COLS = [TARGET_MARK_DOW, TARGET_MARK_DOM, TARGET_MARK_DOY]
VALUE_COLS = ["baidu_x", "baidu_x__missing", "px_a", "rzrq_y"]
FULL_SCHEMA = VALUE_COLS + ALL_CAL + [TARGET]  # 13 列


# ---------------------------------------------------------------- 日历语义


def _holiday_calendar():
    # 2024-02 春节：最后交易日 2024-02-08（周四），长假后首个交易日
    # 2024-02-19（周一），自然日间隔 = 11。全部为工作日，无周末干扰。
    return pd.DatetimeIndex(pd.to_datetime([
        "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08",
        "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22", "2024-02-23",
        "2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29",
        "2024-03-01", "2024-03-04",
    ]))


def _feats(calendar=None, columns=None):
    calendar = _holiday_calendar() if calendar is None else calendar
    columns = ALL_CAL if columns is None else columns
    index = calendar[:-1]
    return calendar, target_calendar_features(calendar, index, columns)


def test_gap_friday_to_monday_is_3():
    calendar, feats = _feats()
    pos = calendar.get_loc(pd.Timestamp("2024-02-23"))  # 周五，目标日=周一
    assert feats[TARGET_GAP_DAYS_LOG][pos] == pytest.approx(np.log1p(3))
    assert feats[TARGET_IS_GAP_GE4][pos] == 0.0
    # 普通相邻交易日 gap=1
    pos = calendar.get_loc(pd.Timestamp("2024-02-20"))
    assert feats[TARGET_GAP_DAYS_LOG][pos] == pytest.approx(np.log1p(1))


def test_long_holiday_post_and_pre_flags_on_different_rows():
    calendar, feats = _feats()

    def at(date, col):
        return feats[col][calendar.get_loc(pd.Timestamp(date))]

    # 行 2024-02-08：目标日 = 2024-02-19（长假后首个交易日），gap = 11
    assert at("2024-02-08", TARGET_GAP_DAYS_LOG) == pytest.approx(np.log1p(11))
    assert at("2024-02-08", TARGET_IS_GAP_GE4) == 1.0
    assert at("2024-02-08", TARGET_IS_PRE_LONG_HOLIDAY) == 0.0

    # 行 2024-02-07：目标日 = 2024-02-08（长假前最后一个交易日）→ pre=1。
    # 与上一行的 post 旗标不在同一行。
    assert at("2024-02-07", TARGET_IS_PRE_LONG_HOLIDAY) == 1.0
    assert at("2024-02-07", TARGET_IS_GAP_GE4) == 0.0

    # 长假后首个交易日本身不是"长假前最后交易日"（其后间隔=1）
    assert at("2024-02-19", TARGET_IS_PRE_LONG_HOLIDAY) == 0.0

    # 目标日为日历末端（2024-03-04）时无法判定其后间隔 → pre=0
    assert at("2024-03-01", TARGET_IS_PRE_LONG_HOLIDAY) == 0.0


def test_month_start_end_3_boundaries():
    calendar, feats = _feats()

    def at(date, col):
        return feats[col][calendar.get_loc(pd.Timestamp(date))]

    # 2024-02 共 13 个交易日：前 3 = 02-05/06/07，后 3 = 02-27/28/29。
    # 目标日 02-06（月内第 2）→ start=1, end=0
    assert at("2024-02-05", TARGET_IS_MONTH_START_3) == 1.0
    assert at("2024-02-05", TARGET_IS_MONTH_END_3) == 0.0
    # 目标日 02-19（月内第 5）→ 两者皆 0
    assert at("2024-02-08", TARGET_IS_MONTH_START_3) == 0.0
    assert at("2024-02-08", TARGET_IS_MONTH_END_3) == 0.0
    # 目标日 02-27（第 11/13，13-11=2<3）→ end=1
    assert at("2024-02-26", TARGET_IS_MONTH_END_3) == 1.0
    assert at("2024-02-26", TARGET_IS_MONTH_START_3) == 0.0
    # 目标日 02-23（第 9/13，13-9=4）→ end=0
    assert at("2024-02-22", TARGET_IS_MONTH_END_3) == 0.0
    # 2024-03 在日历中只有 2 个交易日（03-01/03-04）：同属前 3 与后 3
    assert at("2024-02-29", TARGET_IS_MONTH_START_3) == 1.0  # 目标 03-01
    assert at("2024-02-29", TARGET_IS_MONTH_END_3) == 1.0
    assert at("2024-03-01", TARGET_IS_MONTH_START_3) == 1.0  # 目标 03-04
    assert at("2024-03-01", TARGET_IS_MONTH_END_3) == 1.0


def test_mark_values_use_target_day_official_formulas():
    calendar, feats = _feats()
    # 行 2024-02-08（周四）：目标日 = 2024-02-19（周一）。
    pos = calendar.get_loc(pd.Timestamp("2024-02-08"))
    target = pd.Timestamp("2024-02-19")
    assert feats[TARGET_MARK_DOW][pos] == pytest.approx(
        target.dayofweek / 6.0 - 0.5
    )
    assert feats[TARGET_MARK_DOM][pos] == pytest.approx(
        (target.day - 1) / 30.0 - 0.5
    )
    assert feats[TARGET_MARK_DOY][pos] == pytest.approx(
        (target.dayofyear - 1) / 365.0 - 0.5
    )
    # 目标日语义而非输入日：输入日 02-08 是周四（dow=3），
    # 若误用输入日会得到 3/6-0.5=0.0；目标日周一应为 -0.5。
    assert feats[TARGET_MARK_DOW][pos] == pytest.approx(-0.5)
    # 逐行核对全部标记值都取自下一交易日。
    next_dates = calendar[1:]
    for i, d in enumerate(next_dates):
        assert feats[TARGET_MARK_DOW][i] == pytest.approx(d.dayofweek / 6 - 0.5)
        assert feats[TARGET_MARK_DOM][i] == pytest.approx((d.day - 1) / 30 - 0.5)
        assert feats[TARGET_MARK_DOY][i] == pytest.approx(
            (d.dayofyear - 1) / 365 - 0.5
        )


def test_enabled_subset_and_errors():
    calendar = _holiday_calendar()
    index = calendar[:-1]
    feats = target_calendar_features(calendar, index, [TARGET_GAP_DAYS_LOG])
    assert list(feats) == [TARGET_GAP_DAYS_LOG]
    assert target_calendar_features(calendar, index, []) == {}
    # index 末日没有下一交易日 → 显式报错
    with pytest.raises(ValueError):
        target_calendar_features(calendar, calendar, ALL_CAL)
    # index 含不在日历中的日期 → 显式报错
    with pytest.raises(ValueError):
        target_calendar_features(
            calendar, pd.DatetimeIndex(["2024-02-10"]), ALL_CAL
        )


def test_parse_calendar_features():
    assert parse_calendar_features("all") == ALL_CAL
    assert parse_calendar_features(None) == ALL_CAL
    assert parse_calendar_features("none") == []
    # 子集按规范顺序返回（标记列在前、旗标列在后），与输入顺序无关
    subset = parse_calendar_features(
        "target_is_month_end_3,target_mark_dow,target_gap_days_log"
    )
    assert subset == [TARGET_MARK_DOW, TARGET_GAP_DAYS_LOG, TARGET_IS_MONTH_END_3]
    with pytest.raises(ValueError):
        parse_calendar_features("target_dow_mon")  # 旧 one-hot 列已删除
    with pytest.raises(ValueError):
        parse_calendar_features("not_a_calendar_column")


# ---------------------------------------------------------------- 预处理


def _fit_frame():
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        "px_close": rng.normal(0, 1, n),
        "px_close__missing": np.zeros(n),
        # 训练期恒为常数的日历列：也不允许被丢弃（补零对齐有歧义）
        TARGET_MARK_DOW: np.full(n, -0.5),
        TARGET_IS_GAP_GE4: np.zeros(n),
        TARGET_GAP_DAYS_LOG: rng.uniform(0, 2, n),
        TARGET: rng.normal(0, 0.01, n),
    }, index=idx)
    samples = pd.DataFrame({
        "split": ["train"] * n,
        "label_valid": [True] * n,
        "start_pos": np.arange(n),
        "end_pos": np.arange(n),
    })
    groups = {
        "px_close": ["px_close", "px_close__missing"],
        TARGET_MARK_DOW: [TARGET_MARK_DOW],
        TARGET_IS_GAP_GE4: [TARGET_IS_GAP_GE4],
        TARGET_GAP_DAYS_LOG: [TARGET_GAP_DAYS_LOG],
    }
    return features, samples, groups


def test_calendar_columns_not_standardized_or_dropped():
    features, samples, groups = _fit_frame()
    prep = fit_preprocessing(features, samples, groups, 0.6, 10)
    for col in (TARGET_MARK_DOW, TARGET_IS_GAP_GE4, TARGET_GAP_DAYS_LOG):
        assert col in prep["feature_columns"]
        # 日历列不参与标准化（保持原值；标记列已归一/为 0/1）
        assert prep["statistics"][col] == {
            "median": 0.0, "mean": 0.0, "scale": 1.0,
        }
        # 常数日历列（target_is_gap_ge4 全 0）也不进 dropped_groups
        assert col not in prep["dropped_groups"]
    # 普通数值列仍走标准化（均值/尺度来自训练行）
    stats = prep["statistics"]["px_close"]
    assert stats["scale"] > 0


# ---------------------------------------------------------------- 装载/混用


def _make_symbol_dir(root, symbol, columns, n_dates=40, seq_len=10):
    directory = root / symbol
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B").to_numpy()
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    matrix = rng.normal(size=(n_dates, len(columns))).astype(np.float32)
    # 日历列写入可识别值：列序号*1000 + 行号，便于断言拆分正确性
    for j, c in enumerate(columns):
        if c.startswith("target_"):
            matrix[:, j] = (j + 1) * 1000.0 + np.arange(n_dates)
    np.save(directory / "feature_matrix.npy", matrix)
    np.save(directory / "dates.npy", dates)

    rows = []
    for split, lo, hi in (
        ("train", 0, 15), ("val", 15, 25), ("test", 25, n_dates - 1)
    ):
        for end in range(lo + seq_len - 1, hi):
            rows.append({
                "start_pos": end - seq_len + 1,
                "end_pos": end,
                "target_pos": end + 1,
                "split": split,
                "label_valid": True,
                "y_scaled": 0.1,
                "y_raw": 0.01,
            })
    pd.DataFrame(rows).to_csv(directory / "samples.csv", index=False)

    prep = {
        "symbol": symbol,
        "feature_columns": columns,
        "config": {
            "seq_len": seq_len,
            "return_type": "simple",
            "baidu_keyword": symbol,
        },
        "train_end": "2024-01-22",
        "val_end": "2024-02-05",
        "split_layout": "train_val_test",
        "sample_selection": {"start": "2023-01-01", "tail": None},
        "calendar": "synthetic",
        "version": "stock_timexer_v9",
        # 与样本 y_raw=0.01 / y_scaled=0.1 自洽：(0.01-0)/0.1=0.1
        "statistics": {TARGET: {"median": 0.0, "mean": 0.0, "scale": 0.1}},
    }
    (directory / "preprocessing.json").write_text(json.dumps(prep))


def _make_data_root(tmp_path, symbols_columns):
    root = tmp_path / "data_root"
    root.mkdir()
    spec = {
        "train_end": "2024-01-22",
        "val_end": "2024-02-05",
        "seq_len": 10,
        "split_layout": "train_val_test",
        "train_start": "2023-01-01",
    }
    (root / "export_config.json").write_text(json.dumps(spec))
    report = []
    for symbol, columns in symbols_columns.items():
        _make_symbol_dir(root, symbol, columns)
        report.append({"symbol": symbol, "status": "ok"})
    pd.DataFrame(report).to_csv(root / "export_results.csv", index=False)
    return root


def test_load_catalog_rejects_mixed_calendar_specs(tmp_path):
    full = VALUE_COLS + ALL_CAL + [TARGET]
    no_cal = VALUE_COLS + [TARGET]
    root = _make_data_root(tmp_path, {"A": full, "B": no_cal})
    with pytest.raises(ValueError, match="日历"):
        load_catalog(root)


def test_load_calendar_calendar_columns_at_schema_tail(tmp_path):
    full = VALUE_COLS + ALL_CAL + [TARGET]
    root = _make_data_root(tmp_path, {"A": full})
    records, schema = load_catalog(root)
    assert schema == FULL_SCHEMA  # 值列(排序) + 日历列(规范序) + TARGET
    tail = schema[-1 - len(ALL_CAL):-1]
    assert tail == ALL_CAL
    assert all(c.startswith("target_") for c in tail)


def _windows(tmp_path, mode):
    full = VALUE_COLS + ALL_CAL + [TARGET]
    root = _make_data_root(tmp_path, {"A": full})
    records, schema = load_catalog(root)
    return PanelWindows(
        records, schema, "train", calendar_mode=mode
    ), schema


def test_panel_calendar_mode_time_marks(tmp_path):
    ds, schema = _windows(tmp_path, "time_marks")
    assert len(ds) > 0
    x, y, sid, target, y_raw, ex_mask, x_mark, head_cal = ds[0]
    S = len(schema)
    # 3 个标记列移出 x、进入 x_mark；旗标列仍留在 x
    assert x.shape == (10, S - 3)
    assert x_mark.shape == (10, 3)
    # ex_mask 只覆盖 x 中的外生 token（不含 TARGET 与移出的 3 个标记列）
    assert ex_mask.shape == (S - 1 - 3,)
    assert ex_mask.dtype == np.bool_
    # x_mark / 旗标列与导出矩阵同源：导出列 j 的值 = (j+1)*1000 + 行号，
    # 日历列位于导出列 4..11（标记列 4..6、旗标列 7..11）。
    positions = ds.parts[0][0]
    st, en, tg = positions[0]
    assert target == int(tg)
    rows = np.arange(st, en + 1)
    for k in range(3):
        expected = (4 + k + 1) * 1000.0 + rows
        assert np.allclose(x_mark[:, k], expected), f"mark col {k}"
    # 旗标列在 x 中紧随 4 个值列（schema 顺序），值同源
    for f, cal_k in enumerate(range(len(MARK_COLS), len(ALL_CAL))):
        expected = (4 + cal_k + 1) * 1000.0 + rows
        assert np.allclose(x[:, 4 + f], expected), f"flag col {f}"
    # head_cal = 目标日星期 one-hot（5 维，独热）
    assert head_cal.shape == (5,)
    dow = pd.Timestamp(ds.records[sid]["dates"][tg]).dayofweek
    assert head_cal[dow] == 1.0 and head_cal.sum() == 1.0
    # 标签契约
    assert y.shape == (1, 1) and y.dtype == np.float32
    assert y_raw == pytest.approx(0.01) and isinstance(y_raw, np.float64)


def test_panel_calendar_mode_columns(tmp_path):
    ds, schema = _windows(tmp_path, "columns")
    x, y, sid, target, y_raw, ex_mask, x_mark, head_cal = ds[0]
    S = len(schema)
    assert x.shape == (10, S)  # 全部日历列留在输入列
    assert x_mark.shape == (10, 0)  # 不使用 x_mark 通道
    assert ex_mask.shape == (S - 1,)


def test_panel_calendar_mode_none(tmp_path):
    ds, schema = _windows(tmp_path, "none")
    x, y, sid, target, y_raw, ex_mask, x_mark, head_cal = ds[0]
    S = len(schema)
    assert x.shape == (10, S - len(ALL_CAL))  # 8 个日历列全部移除
    assert x_mark.shape == (10, 0)
    assert ex_mask.shape == (S - len(ALL_CAL) - 1,)
    # head_cal 与日历列开关无关，恒产出
    assert head_cal.shape == (5,) and head_cal.sum() == 1.0


def test_panel_rejects_unknown_calendar_mode(tmp_path):
    full = VALUE_COLS + ALL_CAL + [TARGET]
    root = _make_data_root(tmp_path, {"A": full})
    records, schema = load_catalog(root)
    with pytest.raises(ValueError, match="calendar_mode"):
        PanelWindows(records, schema, "train", calendar_mode="both")


# ---------------------------------------------------------------- 适配器


def _adapter_args(**overrides):
    base = dict(
        d_model=8, e_layers=1, n_heads=2, patch_len=5,
        dropout=0.0, ex_mask=False,
        calendar_mode="time_marks", head_calendar=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_adapter_model_config_three_modes():
    adapter = get_adapter("TimeXer")
    S = len(FULL_SCHEMA)

    cfg = adapter.model_config(_adapter_args(), 10, FULL_SCHEMA)
    assert cfg["calendar_mode"] == "time_marks"
    assert cfg["enc_in"] == S - 3  # 3 个标记列移入 x_mark
    assert cfg["calendar_mark_dim"] == 3
    assert cfg["calendar_mark_columns"] == MARK_COLS
    assert cfg["calendar_columns"] == ALL_CAL
    assert cfg["head_calendar_dim"] == 5  # 默认开启
    assert adapter.checkpoint_format_for(cfg) == "global_timexer_typed_v2"

    cfg = adapter.model_config(
        _adapter_args(calendar_mode="columns"), 10, FULL_SCHEMA
    )
    assert cfg["enc_in"] == S  # 全部日历列留在输入列
    assert cfg["calendar_mark_dim"] == 0
    assert cfg["calendar_mark_columns"] == []
    assert adapter.checkpoint_format_for(cfg) == "global_timexer_typed_v2"

    cfg = adapter.model_config(
        _adapter_args(calendar_mode="none"), 10, FULL_SCHEMA
    )
    assert cfg["enc_in"] == S - len(ALL_CAL)  # 8 个日历列全部移除
    assert cfg["calendar_mark_dim"] == 0

    cfg = adapter.model_config(_adapter_args(head_calendar=False), 10, FULL_SCHEMA)
    assert cfg["head_calendar_dim"] == 0
    assert adapter.checkpoint_format_for(cfg) == "global_timexer_typed_v1"


def test_adapter_forward_head_contract_and_param_count():
    adapter = get_adapter("TimeXer")
    cfg_on = adapter.model_config(_adapter_args(), 10, FULL_SCHEMA)
    cfg_off = dict(cfg_on, head_calendar_dim=0)
    model_on = adapter.build_model(cfg_on)
    model_off = adapter.build_model(cfg_off)

    # 参数量差异 = head Linear 多出的 5 个输入权重（偏置不变）
    n_on = sum(p.numel() for p in model_on.parameters())
    n_off = sum(p.numel() for p in model_off.parameters())
    assert n_on - n_off == 5

    B = 2
    x = torch.randn(B, 10, cfg_on["enc_in"])
    x_mark = torch.randn(B, 10, 3)
    head_cal = torch.zeros(B, 5)
    head_cal[:, 0] = 1.0

    # head 开启：缺 head_calendar 必须报错
    with pytest.raises(ValueError, match="head_calendar"):
        adapter.forward(model_on, x, x_mark=x_mark)
    with torch.no_grad():
        y = adapter.forward(model_on, x, x_mark=x_mark, head_calendar=head_cal)
    assert y.shape == (B, 1, 1) and torch.isfinite(y).all()

    # head 关闭：不需要 head_calendar，输出形状不变
    with torch.no_grad():
        y0 = adapter.forward(model_off, x, x_mark=x_mark)
    assert y0.shape == (B, 1, 1)

    # ex_mask 维度 = 外生 token 数 = enc_in-1+mark_dim（值列+旗标+标记）
    ex_mask = torch.zeros(B, cfg_on["enc_in"] - 1 + 3, dtype=torch.bool)
    with torch.no_grad():
        ym = adapter.forward(
            model_on, x, ex_mask=ex_mask, x_mark=x_mark,
            head_calendar=head_cal,
        )
    assert ym.shape == (B, 1, 1)


# ---------------------------------------------------------------- 上游等价


def _equiv_configs():
    return argparse.Namespace(
        task_name="long_term_forecast",
        features="MS",
        seq_len=60,
        pred_len=1,
        label_len=0,
        enc_in=6,
        c_out=1,
        d_model=16,
        d_ff=32,
        e_layers=2,
        n_heads=4,
        factor=1,
        dropout=0.0,
        embed="fixed",
        freq="d",
        activation="gelu",
        use_norm=0,
        patch_len=10,
    )


def _build_pair():
    torch.manual_seed(20261103)
    configs = _equiv_configs()
    repo = TimeXer(configs).eval()  # 无 head_calendar_dim → 默认 0
    ref = ref_upstream.Model(configs).eval()
    repo_keys = set(repo.state_dict().keys())
    ref_keys = set(ref.state_dict().keys())
    assert repo_keys == ref_keys, (
        f"state_dict 键不一致：多出 {repo_keys - ref_keys}，"
        f"缺少 {ref_keys - repo_keys}"
    )
    ref.load_state_dict(repo.state_dict())
    return repo, ref


def _max_diff(a, b):
    return float((a - b).abs().max().item())


def test_upstream_equivalence_all_switches_off():
    """开关全关（head_calendar_dim=0、ex_mask=None、mask=None）时，
    本仓库 models/TimeXer.py 与 thuml 上游原版数值完全等价。"""
    repo, ref = _build_pair()
    torch.manual_seed(7)
    x = torch.randn(2, 60, 6)

    with torch.no_grad():
        # 1) 无 x_mark
        d_none = _max_diff(
            repo(x, None, None, None), ref(x, None, None, None)
        )
        # 2) 有 x_mark (B, L, 3)
        mark = torch.randn(2, 60, 3)
        d_mark = _max_diff(
            repo(x, mark, None, None), ref(x, mark, None, None)
        )
        # 3) ex_mask 全 False（上游无此参数；全 False 等价于不屏蔽）
        ex_mask = torch.zeros(2, 5, dtype=torch.bool)
        d_exmask = _max_diff(
            repo(x, None, None, None, ex_mask=ex_mask),
            ref(x, None, None, None),
        )
    print(
        f"\n上游等价实测最大绝对差：无mark={d_none:.3e} "
        f"有mark={d_mark:.3e} ex_mask全False={d_exmask:.3e}"
    )
    assert d_none <= 1e-6
    assert d_mark <= 1e-6
    assert d_exmask <= 1e-6


def test_upstream_equivalence_head_dim_zero_structure():
    """head_calendar_dim=0 时结构与上游逐键一致（权重可互载）。"""
    repo, ref = _build_pair()
    assert repo.head_calendar_dim == 0
    # head Linear 输入维 = d_model*(patch_num+1)，与上游相同
    assert (
        repo.head.linear.in_features == ref.head.linear.in_features
        and repo.head.linear.out_features == ref.head.linear.out_features
    )
    torch.manual_seed(11)
    x = torch.randn(1, 60, 6)
    with torch.no_grad():
        # 显式传 head_calendar=None 与上游输出一致
        d = _max_diff(
            repo(x, None, None, None, ex_mask=None, head_calendar=None),
            ref(x, None, None, None),
        )
    assert d <= 1e-6
    # head_calendar_dim=0 时拒绝 head_calendar（防静默错位）
    with pytest.raises(ValueError):
        repo(x, None, None, None, head_calendar=torch.zeros(1, 5))
