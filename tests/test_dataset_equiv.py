# -*- coding: utf-8 -*-
"""P5 数据层拆分等价性验收（需真实行情库，标记 slow）。

固化两类可重复断言：
  1. 拆分后数据引擎的确定性：同一参数两次导出，核心三件产物
     （feature_matrix.npy / dates.npy / samples.csv）逐字节一致；
  2. 旧 data-root 兼容规则：validate 全过、infer 出 latest_window、
     原地重导被 FileExistsError 拒绝（dataset_script_sha256 变化
     不影响旧目录校验，只禁止原地重导，见根目录 dataset.py 头注）；
  3. 若环境变量 P5_EQ_BASELINE 指向"拆分前旧代码导出的目录"
     （含 601398 子目录），则对基线做逐字节 A/B 对比（拆分前
     HEAD f71203f 的实跑结果见提交说明）。

运行：
    .venv/bin/python -m pytest tests/test_dataset_equiv.py -q -m slow
    # 跳过：-m "not slow"，或导出 P5_EQ_SKIP=1
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DB = REPO.parent / "AutoTS" / "download" / "autots.duckdb"
SYMBOL = "601398"
BASELINE_ENV = "P5_EQ_BASELINE"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("P5_EQ_SKIP") == "1",
        reason="P5_EQ_SKIP=1 显式跳过",
    ),
    pytest.mark.skipif(
        not DB.exists(), reason=f"行情库不存在：{DB}"
    ),
]

# 与拆分前 HEAD f71203f 实跑完全相同的参数
PARAMS = [
    "--symbol", SYMBOL,
    "--train-end", "2023-12-29",
    "--val-end", "2025-06-30",
    "--as-of", "2026-10-14",
    "--seq-len", "60",
]

CORE_FILES = ("feature_matrix.npy", "dates.npy", "samples.csv")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(args):
    proc = subprocess.run(
        [sys.executable, str(REPO / "dataset.py")] + args,
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def _export(out_dir):
    _run(["--mode", "prepare", "--db", str(DB), "--out-dir", str(out_dir)]
         + PARAMS)


def _assert_core_identical(dir_a, dir_b, label):
    a, b = Path(dir_a) / SYMBOL, Path(dir_b) / SYMBOL
    for name in CORE_FILES:
        assert _sha256(a / name) == _sha256(b / name), (
            f"{label}: {name} 不一致"
        )


@pytest.fixture(scope="module")
def stock_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("p5_equiv")
    _export(out)
    return out / SYMBOL


def test_two_exports_are_byte_identical(tmp_path, stock_dir):
    """确定性：同参数重导一次，核心产物逐字节一致。"""
    second = tmp_path / "second"
    _export(second)
    _assert_core_identical(stock_dir.parent, second, "重导")


def test_validate_passes_on_exported_dir(stock_dir):
    """旧目录兼容规则：validate 全过。"""
    _run(["--mode", "validate", "--data-dir", str(stock_dir)])


def test_infer_produces_latest_window(tmp_path, stock_dir):
    """旧目录兼容规则：infer 对既有预处理参数出 latest_window。"""
    out = tmp_path / "infer"
    _run([
        "--mode", "infer",
        "--preprocessor", str(stock_dir / "preprocessing.json"),
        "--out-dir", str(out),
        "--db", str(DB),
    ])
    assert (out / SYMBOL / "latest.npz").exists()


def test_inplace_reexport_rejected(tmp_path, stock_dir):
    """原地重导被拒：口径变更必须换新 data-root。"""
    proc = subprocess.run(
        [sys.executable, str(REPO / "dataset.py"), "--mode", "prepare",
         "--db", str(DB), "--out-dir", str(stock_dir.parent)] + PARAMS,
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "FileExistsError" in proc.stderr


@pytest.mark.skipif(
    not os.environ.get(BASELINE_ENV),
    reason=(
        f"未设置 {BASELINE_ENV}（指向拆分前旧代码导出的目录）；"
        "拆分时已人工实跑 A/B 并记录在提交说明"
    ),
)
def test_ab_vs_pre_split_baseline(stock_dir):
    """A/B 硬门槛：与拆分前旧代码的导出逐字节一致。"""
    baseline = Path(os.environ[BASELINE_ENV])
    assert (baseline / SYMBOL).is_dir(), baseline
    # baseline 本身就是导出根目录（含 SYMBOL 子目录）
    _assert_core_identical(baseline, stock_dir.parent, "A/B")
    # 附属 CSV 与 preprocessing.json：既有键一致（允新增）
    a, b = baseline / SYMBOL, stock_dir
    for name in ("aligned_values.csv", "features_unscaled.csv"):
        assert _sha256(a / name) == _sha256(b / name), name
    pb = json.loads((a / "preprocessing.json").read_text())
    pn = json.loads((b / "preprocessing.json").read_text())
    assert set(pb) <= set(pn)
    for key in pb:
        assert pb[key] == pn[key], key
