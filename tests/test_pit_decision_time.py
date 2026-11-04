# -*- coding: utf-8 -*-
"""PIT 校准锚点测试：周三 2026-03-04 06:00 决策时各源取到的观测日。

口径（2026-11-04 校准，决策时点 08:30 -> 06:00）：
决策时点 = 下一交易日 06:00（美股周二场北京时间约 05:00 收盘之后）。
周三 06:00 做预测时：
  - 价量（A股周二收盘）、宏观（含美股周二场）：全部可用；
  - 百度：只有周一（周二那批要周四 00:00 才可用）；
  - 两融：只有周一（周二那批周三 09:00 才发布）。

锚点：2026-03-04 为周三；03-02（周一）、03-03（周二）均为交易日。
available = observed + lag_days + release_hour（见 stock/data/pit.py）。
"""

import pandas as pd

from stock.data import common
from stock.data.pit import align_one
from stock.training import artifacts

DECISION = pd.Timestamp("2026-03-04 06:00")  # 周三 06:00 决策
MON = pd.Timestamp("2026-03-02")
TUE = pd.Timestamp("2026-03-03")


def _align(source, dates):
    spec = {
        "column": f"{source}_x",
        "policy_key": source,
        "source": source,
        "series": pd.Series(
            [1.0 + i for i in range(len(dates))], index=dates),
        "transform": "raw",
    }
    decision_times = pd.Series([DECISION])
    values, ages, audit = align_one(
        spec, decision_times, common.DEFAULT_POLICIES)
    return values.iloc[0], audit


def _observed_avail(source, dates):
    value, audit = _align(source, dates)
    observed = audit[f"{source}_x__observed"].iloc[0]
    available = audit[f"{source}_x__available"].iloc[0]
    print(f"{source}: observed={observed} available={available} "
          f"(decision={DECISION})")
    return observed, available


def test_default_policies_match_pit_calibration():
    """参数本身固化：防止策略被无意改回旧口径。"""
    p = common.DEFAULT_POLICIES
    assert p["macro"] == {"lag_days": 1, "release_hour": 5,
                          "max_age_days": 10}
    assert p["guzhi"] == {"lag_days": 1, "release_hour": 0,
                          "max_age_days": 10}
    assert p["rzrq"] == {"lag_days": 1, "release_hour": 9,
                         "max_age_days": 5}
    assert p["baidu"] == {"lag_days": 2, "release_hour": 0,
                          "max_age_days": 7}


def test_macro_takes_tuesday():
    """宏观（含美股周二场）周三 05:00 可用，06:00 决策取得到周二。"""
    observed, available = _observed_avail("macro", [MON, TUE])
    assert observed == TUE
    assert available == pd.Timestamp("2026-03-04 05:00")
    assert available <= DECISION


def test_guzhi_takes_tuesday():
    """估值次日 00:00 可用，06:00 决策取得到周二。"""
    observed, available = _observed_avail("guzhi", [MON, TUE])
    assert observed == TUE
    assert available == pd.Timestamp("2026-03-04 00:00")
    assert available <= DECISION


def test_baidu_falls_back_to_monday():
    """百度周二批次周四 00:00 才可用，周三 06:00 只能取周一。"""
    observed, available = _observed_avail("baidu", [MON, TUE])
    assert observed == MON
    assert available == pd.Timestamp("2026-03-04 00:00")
    assert available <= DECISION
    # 周二那批确实不可用（available=03-05 00:00 > 决策时点）
    tue_available = TUE + pd.Timedelta(
        days=common.DEFAULT_POLICIES["baidu"]["lag_days"],
        hours=common.DEFAULT_POLICIES["baidu"]["release_hour"])
    assert tue_available == pd.Timestamp("2026-03-05 00:00") > DECISION


def test_rzrq_falls_back_to_monday():
    """两融周二批次周三 09:00 发布，晚于 06:00 决策，只能取周一。"""
    observed, available = _observed_avail("rzrq", [MON, TUE])
    assert observed == MON
    assert available == pd.Timestamp("2026-03-03 09:00")
    assert available <= DECISION
    tue_available = TUE + pd.Timedelta(
        days=common.DEFAULT_POLICIES["rzrq"]["lag_days"],
        hours=common.DEFAULT_POLICIES["rzrq"]["release_hour"])
    assert tue_available == pd.Timestamp("2026-03-04 09:00") > DECISION


def test_price_volume_tuesday_close_is_final():
    """价量无 PIT 策略（直接行情）：周二 15:00 收盘早于周三 06:00 决策，
    周二价量在决策时点必然已定型。"""
    tue_close = TUE + pd.Timedelta(hours=15)
    assert tue_close < DECISION


def test_prediction_time_default_is_0600():
    """训练产物登记的首个预测时刻默认为 test_start 06:00（Asia/Shanghai）。"""
    ts = artifacts.prediction_time({"test_start": "2026-03-04"})
    assert ts == pd.Timestamp("2026-03-04 06:00", tz="Asia/Shanghai")
