# -*- coding: utf-8 -*-
"""选股胜率指标测试：oracle 上限、逐日基线与跨折 mean/worst 汇总。

覆盖：
- stock.evaluation.daily_hit 的 oracle top-N（按真实收益完美排序的
  理论上限）与既有逐日胜率/基线语义不变；
- floor 单位口径（floor_pct 百分点 ↔ floor_decimal 小数）等价；
- stock.evaluation.walk_forward 的逐折选股汇总与跨折 mean/worst
  （不挑最好折）；
- 无信号日的 NaN 统计在写 JSON（allow_nan=False）前被清洗为 None。
"""

import json

import numpy as np
import pandas as pd

from stock.evaluation.daily_hit import daily_hit
from stock.evaluation.walk_forward import (
    aggregate_fold_selection,
    selection_summary_block,
)
from stock.selection.rules import hybrid_signal


def _frame():
    """两天合成预测：
    2026-01-05（20 股）：预测排序与真实收益完全相反（最差模型）；
      y_true 前 10 只为负、后 10 只为正（i>=10 上涨）。
    2026-01-06（10 股）：预测完美且全部上涨。
    """
    rows = []
    n1 = 20
    for i in range(n1):
        rows.append({
            "symbol": f"A{i:02d}",
            "target_date": "2026-01-05",
            "y_pred": (n1 - i) / 100.0,
            "y_true": (i - 9.5) / 100.0,
        })
    n2 = 10
    for i in range(n2):
        rows.append({
            "symbol": f"B{i:02d}",
            "target_date": "2026-01-06",
            "y_pred": (n2 - i) / 100.0,
            "y_true": (n2 - i) / 100.0,
        })
    return pd.DataFrame(rows)


def test_daily_hit_keeps_semantics_and_adds_oracle():
    frame = _frame()
    daily, stats = daily_hit(frame, q_pct=10.0, floor_pct=0.0)

    assert list(daily["target_date"]) == ["2026-01-05", "2026-01-06"]
    # 第 1 天：前 10% = 2 只（A00/A01），预测最高的两只真实收益最低
    assert list(daily["signals"]) == [2, 1]
    assert list(daily["hits"]) == [0, 1]
    assert list(daily["hit_rate"]) == [0.0, 1.0]
    # 同日同合格集合基线上涨率
    assert list(daily["day_up_base"]) == [0.5, 1.0]
    # oracle：按真实收益排序取同分位预算（2 只 / 1 只）
    assert list(daily["oracle_signals"]) == [2, 1]
    assert list(daily["oracle_hits"]) == [2, 1]
    assert list(daily["oracle_hit_rate"]) == [1.0, 1.0]

    assert stats["avg_daily_hit_rate"] == 0.5
    assert stats["avg_daily_base_up_rate"] == 0.75
    assert stats["avg_oracle_hit_rate"] == 1.0
    # 最差模型日胜率 < 基线 < oracle，上限关系成立
    assert (
        stats["avg_daily_hit_rate"]
        <= stats["avg_daily_base_up_rate"]
        <= stats["avg_oracle_hit_rate"]
    )


def test_oracle_tie_break_is_deterministic():
    # 真实收益并列时按 symbol 升序取，结果可复现
    rows = [
        {"symbol": "Z", "target_date": "2026-01-05",
         "y_pred": 0.03, "y_true": 0.01},
        {"symbol": "A", "target_date": "2026-01-05",
         "y_pred": 0.02, "y_true": 0.01},
        {"symbol": "M", "target_date": "2026-01-05",
         "y_pred": 0.01, "y_true": -0.01},
    ]
    daily, _ = daily_hit(pd.DataFrame(rows), q_pct=34.0, floor_pct=0.0)
    # k = ceil(3 * 0.34) = 2；并列 0.01 取 A（symbol 升序），两只均上涨
    assert daily.iloc[0]["oracle_signals"] == 2
    assert daily.iloc[0]["oracle_hits"] == 2
    assert daily.iloc[0]["oracle_hit_rate"] == 1.0


def test_floor_unit_equivalence_with_hybrid_signal():
    frame = _frame()
    daily, stats = daily_hit(frame, q_pct=10.0, floor_pct=0.5)
    # 逐日复算：floor_pct=0.5（百分点）≡ floor_decimal=0.005（小数）
    for _, row in daily.iterrows():
        part = frame[frame.target_date == row["target_date"]]
        signal = hybrid_signal(
            part["y_pred"], 10.0,
            floor_decimal=0.5 / 100.0, pred_unit="decimal",
            symbol=part["symbol"], eligible=part["symbol"].tolist(),
        )["signal"]
        assert int(signal.sum()) == row["signals"]
    assert stats["floor_pct"] == 0.5


def test_selection_summary_block_per_fold_and_cross_fold():
    frame = _frame()
    # 两天拆成两折（walk-forward 中测试块互不重叠）
    frame = frame.assign(fold_id=np.where(
        frame["target_date"] == "2026-01-05", "fold_000", "fold_001"
    ))
    daily_all, block = selection_summary_block(frame, 10.0, 0.0)

    assert list(daily_all["fold_id"]) == ["fold_000", "fold_001"]
    assert [r["fold_id"] for r in block["per_fold"]] == [
        "fold_000", "fold_001"
    ]
    per_fold = {r["fold_id"]: r for r in block["per_fold"]}
    assert per_fold["fold_000"]["avg_daily_hit_rate"] == 0.0
    assert per_fold["fold_001"]["avg_daily_hit_rate"] == 1.0

    cross = block["cross_fold"]
    # 跨折均值与最差折（不是最好折）
    assert cross["avg_daily_hit_rate"]["mean"] == 0.5
    assert cross["avg_daily_hit_rate"]["worst"] == 0.0
    assert cross["avg_oracle_hit_rate"]["mean"] == 1.0
    assert cross["avg_oracle_hit_rate"]["worst"] == 1.0
    assert cross["avg_daily_base_up_rate"]["worst"] == 0.5


def test_aggregate_fold_selection_nan_and_empty():
    per_fold = [
        {"fold_id": "f0", "avg_daily_hit_rate": 0.6,
         "avg_daily_base_up_rate": 0.5, "avg_oracle_hit_rate": 0.9},
        {"fold_id": "f1", "avg_daily_hit_rate": float("nan"),
         "avg_daily_base_up_rate": 0.4, "avg_oracle_hit_rate": 0.8},
    ]
    cross = aggregate_fold_selection(per_fold)
    # NaN 折被剔除：均值/最差只基于有效折
    assert cross["avg_daily_hit_rate"]["mean"] == 0.6
    assert cross["avg_daily_hit_rate"]["worst"] == 0.6
    assert cross["avg_daily_base_up_rate"]["worst"] == 0.4

    empty = aggregate_fold_selection([])
    assert empty["avg_daily_hit_rate"] == {"mean": None, "worst": None}


def test_block_is_json_safe_without_nan():
    # 无信号日：hit_rate 为 NaN，overall/per_fold 必须清洗为 None
    rows = []
    for i in range(20):
        rows.append({
            "symbol": f"S{i:02d}", "target_date": "2026-01-05",
            "fold_id": "fold_000",
            "y_pred": 0.001,  # 0.1% < floor 0.5% → 全天无信号
            "y_true": (i - 10) / 100.0,
        })
    frame = pd.DataFrame(rows)
    _, block = selection_summary_block(frame, 5.0, 0.5)
    assert block["overall"]["total_signals"] == 0
    assert block["overall"]["pooled_hit_rate"] is None
    assert block["overall"]["avg_daily_hit_rate"] is None
    assert block["per_fold"][0]["avg_daily_hit_rate"] is None
    assert block["cross_fold"]["avg_daily_hit_rate"] == {
        "mean": None, "worst": None
    }
    # allow_nan=False 下可写（与 stock.utils.write_json 一致）
    json.dumps(block, ensure_ascii=False, allow_nan=False)
