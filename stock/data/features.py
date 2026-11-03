"""特征构造：固定变换 + 行情派生 + 辅助来源对齐拼接。"""

import numpy as np
import pandas as pd

from stock.data.common import PRICE_FIELDS
from stock.data.pit import align_one
from stock.data.schema import (
    TARGET,
    TARGET_CALENDAR_EXTRA_NAMES,
    TARGET_GAP_DAYS_LOG,
    TARGET_IS_GAP_GE4,
    TARGET_IS_MONTH_END_3,
    TARGET_IS_MONTH_START_3,
    TARGET_IS_PRE_LONG_HOLIDAY,
    TARGET_MARK_DOY,
    TARGET_MARK_DOM,
    TARGET_MARK_DOW,
    TARGET_MARK_NAMES,
    age_col,
    missing_col,
    target_calendar_columns,
)
from stock.data.targets import target_series

def fixed_transform(values, kind):
    if kind == "log1p":
        return np.log1p(values)
    if kind == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    return values


def target_calendar_features(calendar, index, enabled):
    """目标日日历特征：行 t 存 t 的下一交易日（预测目标日）的属性。

    全部由交易日历推导（禁止"日期 >= 28"、"dayofweek + 1"之类的近似）：
      - target_mark_dow/dom/doy：官方日频三特征（与
        utils/timefeatures.py 的 DayOfWeek/DayOfMonth/DayOfYear 同式），
        但按目标交易日取值（第 t 行放 t+1 个交易日的日历值），避免
        输入日/目标日错位一天；
      - target_gap_days_log：log1p(输入日到目标日的自然日数)，
        周五->周一 = 3，长假后可达 8~10；
      - target_is_gap_ge4：自然间隔 >= 4 天（目标日是长假后首个交易日）；
      - target_is_pre_long_holiday：目标日是长假前最后一个交易日
        （目标日到它下一交易日的间隔 >= 4）；目标日为日历末端时
        无法判定，取 0；
      - target_is_month_start_3 / target_is_month_end_3：目标日是其所在
        月份的前/后 3 个交易日（按完整日历内该月交易日排序）。

    calendar：完整交易日历（必须覆盖 index 每行的下一交易日）；
    index：特征矩阵行索引（calendar 的子集）；
    enabled：启用的日历列名集合（仅返回其中的日历列）。
    返回 {列名: float64 数组}。
    """
    enabled = set(enabled)
    wanted = [c for c in target_calendar_columns() if c in enabled]
    if not wanted:
        return {}

    pos = calendar.get_indexer(index)
    if (pos < 0).any():
        raise ValueError("index 含不在交易日历中的日期")
    if (pos + 1 >= len(calendar)).any():
        raise ValueError("每行都必须存在下一交易日（日历须覆盖 as-of 之后）")

    next_pos = pos + 1
    next_dates = calendar[next_pos]
    gap = (next_dates - index).days.astype(np.float64)

    out = {}
    # 官方归一化写法（日频版）：与 hour/23-0.5 同一套 [-0.5, 0.5] 风格。
    if TARGET_MARK_DOW in enabled:
        out[TARGET_MARK_DOW] = next_dates.dayofweek / 6.0 - 0.5
    if TARGET_MARK_DOM in enabled:
        out[TARGET_MARK_DOM] = (next_dates.day - 1) / 30.0 - 0.5
    if TARGET_MARK_DOY in enabled:
        out[TARGET_MARK_DOY] = (next_dates.dayofyear - 1) / 365.0 - 0.5
    if TARGET_GAP_DAYS_LOG in enabled:
        out[TARGET_GAP_DAYS_LOG] = np.log1p(gap)
    if TARGET_IS_GAP_GE4 in enabled:
        out[TARGET_IS_GAP_GE4] = (gap >= 4).astype(np.float64)
    if TARGET_IS_PRE_LONG_HOLIDAY in enabled:
        pre = np.zeros(len(index), dtype=np.float64)
        has_after = next_pos + 1 < len(calendar)
        if has_after.any():
            gap_after = (
                calendar[next_pos[has_after] + 1] - next_dates[has_after]
            ).days
            pre[has_after] = (gap_after >= 4).astype(np.float64)
        out[TARGET_IS_PRE_LONG_HOLIDAY] = pre
    if (
        TARGET_IS_MONTH_START_3 in enabled
        or TARGET_IS_MONTH_END_3 in enabled
    ):
        periods = calendar.to_period("M")
        base = pd.Series(periods)
        rank = base.groupby(periods).cumcount() + 1
        size = base.groupby(periods).transform("size")
        rank_v = rank.to_numpy()[next_pos]
        size_v = size.to_numpy()[next_pos]
        if TARGET_IS_MONTH_START_3 in enabled:
            out[TARGET_IS_MONTH_START_3] = (rank_v <= 3).astype(np.float64)
        if TARGET_IS_MONTH_END_3 in enabled:
            out[TARGET_IS_MONTH_END_3] = (
                (size_v - rank_v) < 3
            ).astype(np.float64)
    return out




def build_features(prices, calendar, as_of, specs, config):
    if as_of not in calendar:
        raise ValueError("as-of 不在交易日历中")
    pos = calendar.get_loc(as_of)
    if pos + 1 >= len(calendar):
        raise ValueError("日历必须包含 as-of 之后的下一交易日")

    unknown = prices.index.difference(calendar)
    if len(unknown):
        raise ValueError(f"行情日期不在交易日历中：{unknown[:5].tolist()}")

    index = calendar[
        (calendar >= prices.index.min()) & (calendar <= as_of)
    ]
    p = prices.reindex(index).copy()

    next_date = pd.Series(calendar[1:].to_numpy(), index=calendar[:-1])
    decision = (
        next_date.reindex(index)
        + pd.Timedelta(
            hours=config["decision_hour"],
            minutes=config["decision_minute"],
        )
    )

    ohlc = p[["open", "close", "high", "low"]]
    valid_price = (
        ohlc.notna().all(axis=1)
        & ohlc.gt(0).all(axis=1)
        & p["high"].ge(p["low"])
        & p["high"].ge(p[["open", "close"]].max(axis=1))
        & p["low"].le(p[["open", "close"]].min(axis=1))
    )
    p.loc[~valid_price, ["open", "close", "high", "low"]] = np.nan
    p.loc[p["turnover"] < 0, "turnover"] = np.nan
    # 成交量缺失/为负只令特征缺失，不单独判定停牌（零量不等于停牌，
    # 停牌日在腾讯行情中通常整行缺失，由 core_valid 机制处理）。
    p.loc[p["volume"] < 0, "volume"] = np.nan

    prev = p["close"].shift(1)
    simple_return = target_series(p["close"], "simple")
    target = target_series(p["close"], config["return_type"])

    raw = p.rename(columns={c: f"px_{c}" for c in PRICE_FIELDS})

    # 行情派生特征。逐列插入会导致 DataFrame 碎片化，先收集再一次拼接。
    derived = {
        "px_open_gap": p["open"] / prev - 1.0,
        "px_intraday_return": p["close"] / p["open"] - 1.0,
        "px_amplitude": (p["high"] - p["low"]) / prev,
    }
    for window in (5, 20):
        derived[f"px_return_{window}"] = (
            (1 + simple_return)
            .rolling(window, min_periods=window)
            .apply(np.prod, raw=True) - 1
        )
        derived[f"px_volatility_{window}"] = (
            simple_return.rolling(window).std()
        )
        derived[f"px_ma_ratio_{window}"] = (
            p["close"] / p["close"].rolling(window).mean() - 1
        )

    mean_turnover = p["turnover"].rolling(20).mean().replace(0, np.nan)
    derived["px_turnover_relative_20"] = p["turnover"] / mean_turnover - 1

    # groups 用于把数值、缺失标记、数据年龄作为一组筛选。
    groups = {}
    base_columns = {c: raw[c] for c in raw.columns}
    base_columns.update(derived)
    new_columns = dict(derived)
    for col in base_columns:
        new_columns[missing_col(col)] = (
            base_columns[col].isna().astype(float)
        )
        groups[col] = [col, missing_col(col)]

    features = pd.concat(
        [raw, pd.DataFrame(new_columns, index=index)], axis=1
    )
    # 成交量原值单位为股、量级巨大（1e8~1e11），特征侧用 log1p；
    # aligned_values.csv 中 px_volume 仍保留原值。
    features["px_volume"] = np.log1p(features["px_volume"])

    # 目标日日历列（3 个官方时间标记列 + 5 个旗标列）：行 t 的目标日 =
    # 以该行为输入日时的预测目标，即下一交易日（decision 的日期部分）。
    # 日历是纯日历信息，预测时点完全已知，不引入未来数据；样本窗口
    # [start..end] 的目标日 date[end+1] 的属性恰为窗口最后一行（决策行），
    # 窗口内其余各行各自表示"以该行为输入日"的目标日属性（窗口互相重叠，
    # 行级存储无法整窗恒定）。恒不缺失，不生成 __missing / __age_days
    # 伴随列，也不参与标准化与屏蔽。星期 one-hot 不再作为列导出
    # （星期信息由 x_mark 时间标记通道与预测头 one-hot 承载）。
    # 启用集合由 config["calendar_features"] 决定（缺省 = 全部，兼容旧
    # 冻结配置）；不同集合构成不同导出规格，装载期禁止混用。
    enabled_calendar = config.get("calendar_features")
    if enabled_calendar is None:
        enabled_calendar = target_calendar_columns()
    enabled_calendar = set(enabled_calendar)

    for col, values in target_calendar_features(
        calendar, index, enabled_calendar
    ).items():
        features[col] = values
        groups[col] = [col]

    # 辅助来源按各自的可用时间对齐，禁止 bfill。
    audit = pd.DataFrame({"decision_time": decision}, index=index)
    raw_aux, aux_columns, audits = {}, {}, []
    for spec in specs:
        col = spec["column"]
        values, age, part_audit = align_one(
            spec, decision, config["policies"]
        )
        raw_aux[col] = values
        aux_columns[col] = fixed_transform(values, spec["transform"])
        aux_columns[age_col(col)] = age
        aux_columns[missing_col(col)] = values.isna().astype(float)
        groups[col] = [
            col, age_col(col), missing_col(col)
        ]
        audits.append(part_audit)

    if raw_aux:
        raw = pd.concat([raw, pd.DataFrame(raw_aux, index=index)], axis=1)
        features = pd.concat(
            [features, pd.DataFrame(aux_columns, index=index)], axis=1
        )
        audit = pd.concat([audit] + audits, axis=1)

    raw[TARGET] = target
    features[TARGET] = target
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features[[c for c in features if c != TARGET] + [TARGET]]

    # 停牌/无行情日及其后无法计算单日收益率的日期不作为有效核心行。
    core_valid = valid_price & target.notna()
    return raw, features, audit, core_valid, valid_price, groups


