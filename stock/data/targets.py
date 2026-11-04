"""预测目标（标签）的单一事实源。

标签定义曾散落在 dataset.py（公式与有效位）、stock/data/schema.py
（目标列名）和文档中。本模块收敛为：

- TARGET_SPEC：目标名称、预测跨度、定义、单位、价格口径、停牌处理；
- target_series()：由收盘价生成监督目标（与旧内联实现逐值一致）；
- label_valid()：目标日是否具备可评分标签；
- target_spec()：写入 preprocessing.json / manifest.json 的登记内容。

注意：本模块只定义"预测标签"。可执行收益（t+1 开盘成交、手续费、
滑点）属于回测层，不在此处伪装成同一口径。
"""

import numpy as np
import pandas as pd

from stock.data.schema import TARGET

#: 目标定义登记。修改口径必须同时提升 FEATURE_SCHEMA_VERSION 之外的
#: 数据版本号（dataset.VERSION），否则历史产物会被静默混用。
TARGET_SPEC = {
    "name": TARGET,
    "horizon": 1,
    "definition": "next_trading_session_close_to_close",
    "unit": "fraction",
    "price_basis": "adjusted_close",
    "target_session": "next_trading_session",
    "suspension": "excluded; missing/invalid target-day OHLC => label_valid=False",
    "decision_time": "next_trading_session 06:00",
    "executable_return": (
        "not this label; executable backtest enters at target-day open"
    ),
}


def target_series(close, return_type="simple"):
    """按收盘价生成目标序列。

    simple: close(t+1) / close(t) - 1.0（历史默认）；
    log:    log(close(t+1) / close(t))。
    与 dataset.build_features 旧内联实现逐值一致（含 NaN 传播）。
    """
    prev = close.shift(1)
    if return_type == "simple":
        return close / prev - 1.0
    if return_type == "log":
        return np.log(close / prev)
    raise ValueError(f"不支持的 return_type：{return_type}")


def label_valid(core_valid, target_value):
    """目标日是否可评分：输入窗口有效且标签为有限值。"""
    return bool(core_valid and np.isfinite(target_value))


def target_spec(return_type="simple"):
    """写入产物的目标登记（含本次实际使用的收益率口径）。"""
    spec = dict(TARGET_SPEC)
    spec["return_type"] = return_type
    return spec
