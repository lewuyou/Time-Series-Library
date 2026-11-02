"""选股规则（全仓唯一实现）。

原先 predict_daily.py 与 analyze_daily_hit.py 各写一份"分位 + 阈值"
混合规则。现统一为 hybrid_signal，并把历史上隐含的口径显式化：

- 生产链路 pred 用百分数、阈值为 0.5（个百分点）；
- 命中分析 pred 用小数、阈值为 0.5/100；
两者都必须通过参数声明单位，禁止悄悄改换算。legacy 位置参数 floor
仅保留给历史回归测试，语义为"与 pred 同单位、直接比较"。

并列处理：
- 传 symbol 时，按 (pred 降序, symbol 升序) 显式排序，结果可复现；
- 不传 symbol 时保持旧的稳定排序（仅供等价性测试）。
"""

import numpy as np
import pandas as pd


def _validate_floor(floor, floor_decimal, floor_pct):
    given = sum(
        value is not None
        for value in (floor, floor_decimal, floor_pct)
    )
    if given != 1:
        raise TypeError(
            "必须且只能指定 floor / floor_decimal / floor_pct 之一"
        )


def hybrid_signal(
    pred,
    quantile_pct,
    floor=None,
    *,
    floor_decimal=None,
    floor_pct=None,
    pred_unit=None,
    symbol=None,
    eligible=None,
):
    """混合选股规则：预测排名前 quantile_pct% 且预测值 > floor。

    参数
    ----
    pred : pd.Series
        预测值。单位由 pred_unit 声明：生产为 "pct"，命中分析为
        "decimal"。保持输入顺序/索引返回。
    quantile_pct : float
        排名分位（百分数，5 表示前 5%）。分母是 eligible 集合大小。
    floor / floor_decimal / floor_pct : float
        三选一。返回列 floor_pass 以 pred 原单位比较：
        - floor_pct：阈值为百分点（生产 0.5 -> 0.5%）；
        - floor_decimal：阈值为小数（命中分析 0.5/100 -> 0.5%）；
        - floor：legacy，与 pred 同单位直接比较（仅回归测试用）。
    pred_unit : {"pct", "decimal"}，使用 floor_decimal/floor_pct 时必填。
    symbol : pd.Series
        与 pred 同索引的股票代码；提供后启用显式并列排序。
    eligible : iterable
        参与排名的股票集合；不传表示全部参与。非 eligible 行 rank 为
        NaN、signal 为 False，但仍保留在返回表中（便于落库 stale/失败）。

    返回与 pred.index 对齐的 DataFrame：rank / quantile_pass /
    floor_pass / signal / eligible。
    """
    pred = pred.astype(float)
    _validate_floor(floor, floor_decimal, floor_pct)

    if floor is not None:
        threshold = float(floor)
        unit = pred_unit or "legacy"
    else:
        if pred_unit not in ("pct", "decimal"):
            raise ValueError(
                "使用 floor_decimal/floor_pct 时必须声明 pred_unit"
                "（pct 或 decimal）"
            )
        floor_fraction = (
            float(floor_decimal) if floor_decimal is not None
            else float(floor_pct) / 100.0
        )
        threshold = (
            floor_fraction * 100.0 if pred_unit == "pct" else floor_fraction
        )
        unit = pred_unit

    mask = pd.Series(True, index=pred.index)
    if eligible is not None:
        if symbol is None:
            raise ValueError("传入 eligible 时必须同时传入 symbol")
        mask = symbol.reindex(pred.index).isin(set(eligible))
        mask = mask.fillna(False).astype(bool)

    if symbol is not None:
        sym = symbol.reindex(pred.index)
        ordered = (
            pd.DataFrame({"p": pred, "s": sym})[mask]
            .sort_values(
                ["p", "s"], ascending=[False, True], kind="mergesort"
            )
            .index
        )
    else:
        ordered = pred[mask].sort_values(ascending=False).index

    n = int(mask.sum())
    k = int(np.ceil(n * float(quantile_pct) / 100.0))
    rank = pd.Series(np.arange(1, n + 1), index=ordered, dtype="float64")

    result = pd.DataFrame(index=pred.index)
    result["rank"] = rank.reindex(pred.index)
    result["quantile_pass"] = (result["rank"] <= k).fillna(False)
    result["floor_pass"] = pred > threshold
    result["signal"] = (
        result["quantile_pass"] & result["floor_pass"] & mask
    )
    result["eligible"] = mask
    result.attrs["floor_threshold"] = threshold
    result.attrs["pred_unit"] = unit
    return result
