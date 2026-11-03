"""股票池规模回归测试：默认参数必须复现历史口径（约 1908 只）。

形态同 test_daily_pipeline.py：slow + 缺库跳过。数据库为固定快照
（2026-10-14），断言用数量区间而非精确值，容忍采集库的小幅更新。
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DB = REPO.parent / "AutoTS" / "download" / "autots.duckdb"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DB.exists(), reason=f"行情库不存在：{DB}"),
]


def test_default_universe_matches_historical_scale(tmp_path):
    """缺省（top_mv=3000/min_years=3/fresh=30）入选应在历史量级。

    2026-10-14 快照的金标准漏斗（dataset/gseq/universe_meta.json）：
    top_mv 3000 → has_quotes 2971 → span 2797 → quote_days 2796 →
    has_rzrq 2446 → has_fresh_baidu 1908。
    """
    out = tmp_path / "universe.csv"
    subprocess.run([
        sys.executable, "-m", "stock.selection.universe",
        "--db", str(DB), "--out", str(out),
        "--top-mv", "3000", "--min-years", "3", "--baidu-fresh-days", "30",
    ], cwd=REPO, check=True, capture_output=True)

    frame = pd.read_csv(out, dtype={"symbol": str})
    meta = json.loads(
        (tmp_path / "universe_meta.json").read_text(encoding="utf-8")
    )
    stages = meta["stages"]

    # 漏斗单调不增
    order = ["top_mv", "has_quotes", "span_ge_1095d",
             "quote_days_ge_732", "has_rzrq", "has_fresh_baidu"]
    values = [stages[key] for key in order]
    assert values == sorted(values, reverse=True)
    assert stages["top_mv"] == 3000

    # 各阶段数量区间（金标准 ±10%，防回归把条件悄悄收紧/放宽）
    assert 2600 <= stages["has_quotes"] <= 3000
    assert 2400 <= stages["span_ge_1095d"] <= 3000
    assert 2100 <= stages["has_rzrq"] <= 2700
    assert 1600 <= stages["has_fresh_baidu"] <= 2100

    assert meta["selected"] == len(frame)
    assert 1600 <= len(frame) <= 2100
    # 入选列齐全且无重复
    assert not frame["symbol"].duplicated().any()
    for col in ("symbol", "mv_rank", "total_mv", "baidu_last", "rzrq_rows"):
        assert col in frame.columns
