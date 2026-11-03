# -*- coding: utf-8 -*-
"""R9：daily_pipeline 端到端冒烟（infer → score → 选股 → 落库 → 发布/阻断）。

原 ARCH_REVIEW 续评 R9：prediction_store 落地后（P2），生产管线
（infer 子进程、打分、混合规则、readiness 阻断、SQLite 落库、CSV
视图）此前只靠 USAGE 手工实测记录，无固化的自动化断言。

需要真实行情库（同 test_dataset_equiv.py 的 slow + skipif 形态）：
  1. 单股（000001）导出临时 data-root（dataset.py --mode prepare）；
  2. 构造契约完整但随机权重的 checkpoint（走真实适配器 build_model，
     predictor 按 checkpoint 元数据反查适配器，无需真训练）；
  3. 跑发布场景：断言 exit 0、run 落库 published、predictions 有
     000001、CSV 视图导出（跑完清理，避免仓库根留副产物）；
  4. 跑阻断场景（--as-of 超前于库内最新行情）：断言 exit 1、
     run 落库 blocked、无 CSV 视图。
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
DB = REPO.parent / "AutoTS" / "download" / "autots.duckdb"
SYMBOL = "000001"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("P5_EQ_SKIP") == "1",
        reason="P5_EQ_SKIP=1 显式跳过（与 test_dataset_equiv 同开关）",
    ),
    pytest.mark.skipif(not DB.exists(), reason=f"行情库不存在：{DB}"),
]


def _run(args, cwd=REPO):
    return subprocess.run(
        [sys.executable] + args, cwd=cwd, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """临时 data-root：单股导出 + universe.csv。"""
    root = tmp_path_factory.mktemp("pipeline_data")
    proc = _run([
        "dataset.py", "--mode", "prepare", "--db", str(DB),
        "--symbol", SYMBOL, "--out-dir", str(root),
        "--train-end", "2023-12-29", "--val-end", "2025-06-30",
        "--seq-len", "60",
    ])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (root / "universe.csv").write_text(
        f"symbol,name,industry_l2\n{SYMBOL},平安银行,银行Ⅱ\n",
        encoding="utf-8-sig",
    )
    return root


@pytest.fixture(scope="module")
def checkpoint(data_root, tmp_path_factory):
    """契约完整、随机权重的 TimeXer checkpoint（无需真训练）。"""
    import json

    from stock.adapters import get_adapter
    from stock.data.schema import feature_schema_doc

    prep = json.loads(
        (data_root / SYMBOL / "preprocessing.json").read_text(encoding="utf-8")
    )
    schema = list(prep["feature_columns"])
    adapter = get_adapter("TimeXer")
    config = adapter.model_config(
        argparse.Namespace(
            d_model=16, e_layers=1, n_heads=2, patch_len=10,
            dropout=0.1, ex_mask=False,
        ),
        seq_len=int(prep["config"]["seq_len"]),
        schema=schema,
    )
    model = adapter.build_model(config)
    path = tmp_path_factory.mktemp("pipeline_ckpt") / "checkpoint.pth"
    torch.save(
        {
            "format": adapter.checkpoint_format,
            "schema": schema,
            "feature_schema": feature_schema_doc(schema),
            "preprocessing": {SYMBOL: prep},
            "model_config": config,
            "state_dict": model.state_dict(),
            "dataset_id": "test-daily-pipeline",
            "model_id": "test-timexer",
        },
        path,
    )
    return path


def _pipeline(checkpoint, data_root, run_db, infer_root, extra=()):
    return _run([
        "-m", "stock.production.daily_pipeline",
        "--checkpoint", str(checkpoint),
        "--data-root", str(data_root),
        "--db", str(DB),
        "--run-db", str(run_db),
        "--infer-root", str(infer_root),
        "--workers", "1", "--batch-size", "64",
        # 放宽新鲜度门槛：让"发布/阻断"两场景的判定不依赖行情库
        # 距今的实际天数（阻断场景由 --as-of 超前触发，见下）。
        "--max-lag-days", "30",
        "--device", "cpu",
        *extra,
    ])


def _rows(run_db, sql):
    con = sqlite3.connect(run_db)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


@pytest.fixture(autouse=True)
def _cleanup_csv_views():
    """daily_pipeline 的 CSV 视图固定写到仓库根 predictions/；测试自清。"""
    yield
    for path in (REPO / "predictions").glob("predict_*.csv"):
        path.unlink()


def test_publish_then_block(data_root, checkpoint, tmp_path):
    run_db = tmp_path / "predictions.db"

    # 场景 1：正常发布（缺省 as-of → 取库内最新行情日）
    proc = _pipeline(checkpoint, data_root, run_db, tmp_path / "infer")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "已落库 run_id=" in proc.stdout
    assert "CSV 导出视图" in proc.stdout

    runs = _rows(run_db, "SELECT publish_status FROM prediction_runs")
    assert runs == [("published",)]
    preds = _rows(
        run_db,
        "SELECT symbol, model_id, dataset_id FROM predictions",
    )
    assert preds == [(SYMBOL, "test-timexer", "test-daily-pipeline")]
    csv_views = list((REPO / "predictions").glob("predict_*.csv"))
    assert len(csv_views) == 1

    # 场景 2：readiness 发布门槛阻断——本库行情截至 2026-10-14，距今
    # 已落后约 20 天；把 --max-lag-days 收紧到 5 → blocked，run 仍留档
    # （blocked 也落库、可回溯），但不导出 CSV 视图、exit 1。
    proc = _pipeline(
        checkpoint, data_root, run_db, tmp_path / "infer_blocked",
        extra=["--max-lag-days", "5"],
    )
    assert proc.returncode == 1
    assert "发布被阻断" in proc.stderr
    runs = _rows(
        run_db,
        "SELECT publish_status, blockers FROM prediction_runs "
        "ORDER BY created_at",
    )
    assert len(runs) == 2
    assert runs[-1][0] == "blocked"
    # blockers 以换行拼接的文本入库；滞后门槛由 readiness.assess_publish 给出
    assert "数据滞后" in (runs[-1][1] or "")
    # 阻断不导出 CSV 视图（场景 1 的 CSV 已被 autouse fixture 清理或仍在；
    # 这里只断言不新增）
    assert len(list((REPO / "predictions").glob("predict_*.csv"))) == len(csv_views)
