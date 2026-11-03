# -*- coding: utf-8 -*-
"""拆分等价性与契约回归测试（由 tsl_refactor/verify_equiv.py 迁入仓库）。

参照物是拆分前各文件中的原始实现（predict_daily.py /
analyze_daily_hit.py / train_global_timexer.py 行 141-193 /
walk_forward.py date_text / dataset.py json_write / export_global.py）。
本文件把旧实现逐字复制为纯函数或以内联闭包形式保留，不读取仓库外的
orig/ 目录；export.py 的五个入口函数改为"源码哈希冻结"：哈希以当前
远程版本为准（2026-11-02），函数体一旦变化即失败，提醒同步更新导出
配置版本，而不是静默放行。

运行：仓库根目录 `.venv/bin/python -m pytest tests -q`
"""

import hashlib
import importlib
import inspect
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stock.data import schema
from stock.selection.rules import hybrid_signal
from stock.utils import date_text, read_json, write_json

# export.py 五个入口函数的源码哈希基线（以当前远程版本为准）。
# 这些函数体被 export_config.json 的 sha 校验与历史 data-root 复用依赖，
# 任何改动都必须伴随新的导出目录，不能静默修改。
# 基线更新（P6 日历特征）：parse_args/init_worker/main 因新增
# --calendar-features 参数而变化；resolve_split/export_one 未改动。
# 旧 data-root 的 export_config.json 记录了旧哈希，复用校验会显式失败，
# 符合"日历列改变 schema 必须重导"的约束。
EXPORT_FUNCTION_SHA256 = {
    "parse_args": "0683093509b8cd97db4a02e063f37ce22cf1dc3bcf8618d13155339bed03e91e",
    "resolve_split": "185c4aa1bb4a1adf873aba73ba916caa23742942de565b05c5fbb1c3081fec22",
    "init_worker": "3e2be9d28574e000804ebff2787a33487fbfe469fa53218e0afef822674cbbf9",
    "export_one": "f4494cac09b5a7e4f30309af12a1a25c636477d03c361a8770b737b6b4392b0e",
    "main": "7771888cd856d55cdcf27dc2a3628487aeed633b2344cfd395c4c61ac8caddb3",
}


# ------------------------------------------------ 旧实现（逐字复制，参照物）

def old_base_name(column):
    return column.split("__", 1)[0]


def old_align_features(x, source_columns, schema_cols):
    result = np.zeros((*x.shape[:-1], len(schema_cols)), dtype=np.float32)
    missing_positions = [
        i for i, c in enumerate(schema_cols) if c.endswith("__missing")
    ]
    result[..., missing_positions] = 1.0

    source = {c: i for i, c in enumerate(source_columns)}
    dst = [i for i, c in enumerate(schema_cols) if c in source]
    src = [source[schema_cols[i]] for i in dst]
    result[..., dst] = x[..., src]
    return result


def old_exogenous_mask_layout(schema_cols):
    pos = {c: i for i, c in enumerate(schema_cols)}
    pairs = []
    for j, c in enumerate(schema_cols[:-1]):
        if c.endswith("__missing") or c.endswith("__age_days"):
            continue
        flag = pos.get(c + "__missing")
        if flag is not None:
            pairs.append((j, flag))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def old_compute_ex_mask(x, layout):
    ex_mask = np.zeros(x.shape[:-2] + (x.shape[-1] - 1,), dtype=bool)
    if len(layout):
        masked = (x[..., layout[:, 1]] == 1.0).all(axis=-2)
        ex_mask[..., layout[:, 0]] = masked
    return ex_mask


def old_date_text(value):
    return str(pd.Timestamp(value).date())


def old_json_write(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def old_predict_daily_rule(valid, q_pct, floor_pct):
    """predict_daily.py 行 217-224 的内联规则。"""
    valid = valid.sort_values("pred_return_pct", ascending=False)
    n = len(valid)
    k = int(np.ceil(n * q_pct / 100.0))
    out = pd.DataFrame(index=valid.index)
    out["rank"] = np.arange(1, n + 1)
    out["quantile_pass"] = out["rank"] <= k
    out["floor_pass"] = valid.pred_return_pct > floor_pct
    out["signal"] = out.quantile_pass & out.floor_pass
    return out


def old_daily_hit_signals(frame, q_pct, floor_pct):
    """analyze_daily_hit.py daily_hit 的逐日信号部分（小数口径）。"""
    out = {}
    for date, part in frame.groupby("target_date"):
        part = part.sort_values("y_pred", ascending=False)
        n = len(part)
        k = int(np.ceil(n * q_pct / 100.0))
        rank = np.arange(1, n + 1)
        signal = (rank <= k) & (part.y_pred > floor_pct / 100.0)
        out[date] = signal
    return out


# ---------------------------------------------------------------- 测试主体

def test_hybrid_signal_matches_old_predict_rule():
    rng = np.random.default_rng(20261102)
    for trial in range(30):
        n = int(rng.integers(1, 60))
        preds = np.round(rng.normal(0.3, 1.5, n), 3)
        preds[rng.random(n) < 0.1] = 0.5  # 制造并列与 floor 边界
        q = float(rng.choice([1, 5, 10, 20, 33.3]))
        floor = float(rng.choice([0.0, 0.3, 0.5, 1.0]))
        valid = pd.DataFrame({"pred_return_pct": preds})
        old = old_predict_daily_rule(valid, q, floor)
        # 生产口径：pred 为百分数，floor_pct 同为百分数；不传 symbol 时
        # 保持旧的稳定排序语义（等价性对照必须逐位一致）。
        new = hybrid_signal(
            valid["pred_return_pct"], q, floor,
        ).loc[old.index]
        for column in ("rank", "quantile_pass", "floor_pass", "signal"):
            assert np.array_equal(
                old[column].to_numpy(), new[column].to_numpy()
            ), f"hybrid_signal {column} trial={trial}"

        # 生产调用路径（先排序 reset 再调用，位置对齐）
        vs = valid.sort_values(
            "pred_return_pct", ascending=False
        ).reset_index(drop=True)
        prod = hybrid_signal(
            vs["pred_return_pct"], q, floor,
        )
        assert np.array_equal(prod["rank"].to_numpy(), old["rank"].to_numpy()), (
            f"hybrid_signal 生产路径 trial={trial}"
        )


def test_hybrid_signal_matches_old_daily_hit():
    rng = np.random.default_rng(20261102)
    dates = pd.date_range("2024-01-02", periods=12)
    rows = []
    for d in dates:
        n = int(rng.integers(3, 40))
        rows.append(pd.DataFrame({
            "target_date": d,
            "y_pred": np.round(rng.normal(0.002, 0.01, n), 5),
        }))
    frame = pd.concat(rows, ignore_index=True)
    for q, floor in [(5.0, 0.5), (10.0, 0.2), (1.0, 0.0)]:
        old_sig = old_daily_hit_signals(frame, q, floor)
        for date, part in frame.groupby("target_date"):
            # 命中分析口径：pred 为小数，floor 换算为 0.5/100。
            new = hybrid_signal(
                part["y_pred"], q, floor / 100.0,
            )["signal"]
            part_sorted = part.sort_values("y_pred", ascending=False)
            assert np.array_equal(
                old_sig[date].to_numpy(),
                new.reindex(part_sorted.index).to_numpy(),
            ), f"daily_hit {date} q={q} floor={floor}"


def test_schema_alignment_matches_old_inline():
    rng = np.random.default_rng(20261102)
    schema_cols = (
        ["px_close_ret", "px_volume_log"]
        + ["target_dow_" + d for d in ("mon", "tue", "wed", "thu", "fri")]
        + [
            "baidu_search_all", "baidu_search_all__missing",
            "baidu_search_all__age_days",
            "val_pe", "val_pe__missing",
            "macro_SSE:000001_close", "macro_SSE:000001_close__missing",
            "macro_SSE:000001_volume",
            "rzrq_fin_tval_ratio", "rzrq_fin_tval_ratio__missing",
            schema.TARGET,
        ]
    )
    for trial in range(20):
        keep = [c for c in schema_cols if rng.random() > 0.2]
        if not keep:
            keep = schema_cols[:1]
        x = rng.normal(size=(4, 30, len(keep))).astype(np.float32)
        keep_perm = list(keep)
        rng.shuffle(keep_perm)
        perm = np.array([keep.index(c) for c in keep_perm])
        a_old = old_align_features(x[:, :, perm], keep_perm, schema_cols)
        a_new = schema.align_features(x[:, :, perm], keep_perm, schema_cols)
        assert np.array_equal(a_old, a_new), f"align_features trial={trial}"
        lo = old_exogenous_mask_layout(schema_cols)
        ln = schema.exogenous_mask_layout(schema_cols)
        assert np.array_equal(lo, ln) and lo.dtype == ln.dtype
        mo = old_compute_ex_mask(a_old, lo)
        mn = schema.compute_ex_mask(a_new, ln)
        assert np.array_equal(mo, mn), f"compute_ex_mask trial={trial}"

    a_all_missing = schema.align_features(
        rng.normal(size=(2, 30, 1)).astype(np.float32),
        ["only_col"], schema_cols,
    )
    m = schema.compute_ex_mask(
        a_all_missing, schema.exogenous_mask_layout(schema_cols)
    )
    lo = old_compute_ex_mask(
        a_all_missing, old_exogenous_mask_layout(schema_cols)
    )
    assert np.array_equal(m, lo)
    for c in ["baidu_x__missing", "macro_y__age_days", "target_dow_mon",
              "val_pe", "a__b__missing"]:
        assert old_base_name(c) == schema.base_name(c)


def test_date_text_and_json_writer():
    samples = [
        "2024-02-29", "2016-01-01", "2023-12-31 23:59:59",
        pd.Timestamp("2020-06-30"), np.datetime64("2021-03-14"),
        "2024-03-05",
    ]
    assert all(date_text(v) == old_date_text(v) for v in samples)

    obj = {"中文": "键值", "n": 3, "pi": 3.14, "nested": {"a": [1, 2, None]}}
    with tempfile.TemporaryDirectory() as td:
        p_old = Path(td) / "old.json"
        p_new = Path(td) / "new.json"
        old_json_write(p_old, obj)
        write_json(p_new, obj)
        assert p_old.read_bytes() == p_new.read_bytes()
        assert read_json(p_new) == obj
        assert not (Path(td) / "new.json.tmp").exists()


def test_module_symbol_bindings():
    import dataset as ds
    import stock.data.schema as schema_mod
    import stock.evaluation.metrics as metrics_mod
    import stock.evaluation.walk_forward as wf

    assert ds.json_write is write_json
    assert ds.TARGET == schema.TARGET
    assert ds.missing_col("a") == "a__missing"
    assert ds.age_col("a") == "a__age_days"
    # target_dow_* one-hot 已删除（P6 新设计）：星期信息由 x_mark
    # 时间标记通道与预测头 one-hot 承载，dow_column/is_dow_col 不复存在。
    assert ds.is_exogenous_col("baidu_x")
    assert not ds.is_exogenous_col("px_close")
    assert schema_mod.align_features is schema.align_features
    assert schema_mod.exogenous_mask_layout is schema.exogenous_mask_layout
    assert schema_mod.compute_ex_mask is schema.compute_ex_mask
    assert callable(metrics_mod.save_test_report)
    assert wf.date_text is date_text
    assert importlib.import_module("stock.production.daily_pipeline")
    assert importlib.import_module("stock.evaluation.daily_hit")


def test_export_functions_frozen_and_module_entry():
    new_export = importlib.import_module("stock.data.export")
    for name, expected in EXPORT_FUNCTION_SHA256.items():
        source = inspect.getsource(getattr(new_export, name))
        actual = hashlib.sha256(source.encode()).hexdigest()
        assert actual == expected, (
            f"stock.data.export.{name} 源码已变化（{actual}）；"
            "必须换新的 data-root 并更新本测试基线"
        )

    pickle.loads(pickle.dumps(new_export.export_one))
    pickle.loads(pickle.dumps(new_export.init_worker))

    proc = subprocess.run(
        [sys.executable, "-m", "stock.data.export", "--help"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0 and "--universe-csv" in proc.stdout
