"""特征构造：固定变换 + 行情派生 + 辅助来源对齐拼接。"""

import numpy as np
import pandas as pd

from stock.data.common import PRICE_FIELDS
from stock.data.pit import align_one
from stock.data.schema import (
    TARGET,
    TARGET_DOW_NAMES,
    age_col,
    dow_column,
    missing_col,
)
from stock.data.targets import target_series

def fixed_transform(values, kind):
    if kind == "log1p":
        return np.log1p(values)
    if kind == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    return values




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

    # 目标日星期 one-hot（周一~周五）：行 t 的目标日 = 以该行为输入日
    # 时的预测目标，即下一交易日（decision 的日期部分）。星期是纯日历
    # 信息，预测时点完全已知，不引入未来数据；样本窗口 [start..end] 的
    # 目标日 date[end+1] 的 one-hot 恰为窗口最后一行（决策行），窗口内
    # 其余各行各自表示"以该行为输入日"的目标日星期（窗口互相重叠，
    # 行级存储无法整窗恒定）。交易日只落在周一~周五，5 列完整覆盖；
    # 恒不缺失，不生成 __missing / __age_days 伴随列，也不参与屏蔽。
    target_dow = next_date.reindex(index).dt.dayofweek
    for i, name in enumerate(TARGET_DOW_NAMES):
        col = dow_column(name)
        features[col] = (target_dow == i).astype(float)
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


