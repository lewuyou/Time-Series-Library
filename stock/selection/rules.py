"""选股规则（全仓唯一实现）。

原先 predict_daily.py 与 analyze_daily_hit.py 各写一份"分位 + 阈值"
混合规则，参数口径（百分数/小数）还不同，改动需同步两处。现统一为
hybrid_signal：调用方负责把 floor 换算成与预测值相同的单位。
"""

import numpy as np
import pandas as pd


def hybrid_signal(pred, quantile_pct, floor):
    """混合选股规则：预测排名前 quantile_pct% 且预测值 > floor。

    pred: pd.Series 预测值（任意顺序，保持原索引返回）；
    quantile_pct: 排名分位（百分数，如 5 表示前 5%）；
    floor: 预测值绝对下限，与 pred 同单位（pct 或小数由调用方统一）。
    返回与 pred.index 对齐的 DataFrame：rank / quantile_pass /
    floor_pass / signal。排名语义与原实现逐位一致：按 pred 降序
    sort_values 后顺序编号（1 起）。
    """
    pred = pred.astype(float)
    ordered = pred.sort_values(ascending=False)
    n = len(pred)
    k = int(np.ceil(n * float(quantile_pct) / 100.0))
    rank = pd.Series(np.arange(1, n + 1), index=ordered.index)

    result = pd.DataFrame(index=pred.index)
    result["rank"] = rank.reindex(pred.index)
    result["quantile_pass"] = result["rank"] <= k
    result["floor_pass"] = pred > floor
    result["signal"] = result["quantile_pass"] & result["floor_pass"]
    return result
