"""Point-in-time 对齐与数据修订审计（自 dataset.py 原样迁出）。"""

from pathlib import Path

import numpy as np
import pandas as pd

from stock.data.common import sql_symbol

def align_one(spec, decision_times, policies):
    """按假设可用时间匹配最新非空值，过期则置空。"""
    col = spec["column"]
    policy = policies.get(spec["policy_key"], policies[spec["source"]])
    series = spec["series"].dropna().sort_index()
    series = series[np.isfinite(series.to_numpy(dtype=float))]

    if spec["transform"] == "log1p" and (series < 0).any():
        raise ValueError(f"{col} 存在不允许的负值")

    index = decision_times.index
    if series.empty:
        return (
            pd.Series(np.nan, index=index),
            pd.Series(np.nan, index=index),
            pd.DataFrame({
                f"{col}__observed": pd.NaT,
                f"{col}__available": pd.NaT,
            }, index=index),
        )

    right = pd.DataFrame({
        "observed": series.index,
        "value": series.to_numpy(dtype=float),
    })
    right["available"] = (
        right["observed"]
        + pd.Timedelta(days=policy["lag_days"])
        + pd.Timedelta(hours=policy["release_hour"])
    )

    matched = pd.merge_asof(
        pd.DataFrame({"decision": decision_times.to_numpy()}),
        right.sort_values("available"),
        left_on="decision",
        right_on="available",
        direction="backward",
    )

    age = (
        matched["decision"] - matched["observed"]
    ).dt.total_seconds() / 86400.0

    usable = (
        matched["value"].notna()
        & age.ge(0)
        & age.le(policy["max_age_days"])
    )
    values = pd.Series(
        matched["value"].where(usable).to_numpy(), index=index
    )
    ages = pd.Series(age.to_numpy(), index=index)
    audit = pd.DataFrame({
        f"{col}__observed": matched["observed"].to_numpy(),
        f"{col}__available": matched["available"].to_numpy(),
    }, index=index)

    known = matched["available"].notna()
    if (
        matched.loc[known, "available"] > matched.loc[known, "decision"]
    ).any():
        raise AssertionError("发现未来可用数据")

    return values, ages, audit




def source_watermarks(con, symbol, keyword):
    """各来源当前入库的最大观测日期（信息水位，不代表真实发布时间）。"""

    def q(sql, params):
        value = con.execute(sql, params).fetchone()
        if not value or value[0] is None:
            return None
        return str(pd.Timestamp(value[0]).date())

    return {
        "akshare_tx": q(
            f"SELECT max(date) FROM akshare_tx "
            f"WHERE {sql_symbol('symbol')} = ? AND adjust = 'hfq'",
            [symbol],
        ),
        "baidu": q(
            "SELECT max(date) FROM baidu WHERE keyword = ?", [keyword]
        ),
        "guzhi": q(
            f"SELECT max(date) FROM guzhi WHERE {sql_symbol('symbol')} = ?",
            [symbol],
        ),
        "rzrq": q(
            f"SELECT max(date) FROM rzrq WHERE {sql_symbol('code')} = ?",
            [symbol],
        ),
        "tradingview_macro": q(
            "SELECT max(date) FROM tradingview_macro", []
        ),
    }




def history_drift_report(preprocessor_path, matrix, features, columns):
    """比对 infer 重算的历史矩阵与训练期导出矩阵。

    infer 只冻结预处理参数，历史行按当前数据库重算；若数据库历史值被
    修订（如 TradingView 全历史重抓覆盖），相同日期的特征行会变化。
    返回逐行漂移统计，供 manifest 与告警使用。

    columns 为矩阵的列名（= 预处理 feature_columns）；features.columns
    包含被覆盖率过滤掉的列，不能用于矩阵列名映射。
    """
    base = Path(preprocessor_path).parent
    matrix_path, dates_path = base / "feature_matrix.npy", base / "dates.npy"
    if not (matrix_path.exists() and dates_path.exists()):
        return {"checked": False, "reason": "训练目录缺少矩阵文件，无法比对"}

    old_matrix = np.load(matrix_path)
    old_dates = pd.DatetimeIndex(np.load(dates_path))
    if old_matrix.shape[1] != matrix.shape[1] or len(columns) != matrix.shape[1]:
        return {
            "checked": False,
            "reason": (
                f"特征列数不同：训练期 {old_matrix.shape[1]} vs 当前 "
                f"{matrix.shape[1]}"
            ),
        }

    new_dates = features.index
    common = old_dates.intersection(new_dates)
    if len(common) == 0:
        return {"checked": False, "reason": "无重叠日期"}

    old_pos = old_dates.get_indexer(common)
    new_pos = new_dates.get_indexer(common)
    delta = np.abs(
        old_matrix[old_pos].astype(np.float64) - matrix[new_pos].astype(np.float64)
    )
    changed_rows = np.flatnonzero((delta > 1e-9).any(axis=1))
    changed_columns = [
        columns[i] for i in np.flatnonzero((delta > 1e-9).any(axis=0))
    ]
    return {
        "checked": True,
        "compared_rows": int(len(common)),
        "changed_rows": int(len(changed_rows)),
        "changed_row_dates": [
            str(new_dates[new_pos[i]].date()) for i in changed_rows[:20]
        ],
        "changed_columns": changed_columns,
        "max_abs_diff": float(delta.max()) if delta.size else 0.0,
    }


