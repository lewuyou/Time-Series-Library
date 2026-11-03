"""预处理拟合与变换（fit_preprocessing / transform）。"""

import numpy as np
import pandas as pd

from stock.data.schema import TARGET, is_calendar_col, is_missing_col

def fit_preprocessing(features, samples, groups, min_coverage, min_obs):
    # 只用合法训练样本（label_valid）拟合：无标签的"可预测"行不参与
    # 统计量估计，也不贡献 fit 行。
    train = samples[
        (samples["split"] == "train") & samples["label_valid"]
    ]
    if train.empty:
        raise ValueError("没有有效训练样本")

    # 差分数组构造 fit mask：O(样本数 + 行数)，避免逐窗口切片赋值。
    diff = np.zeros(len(features) + 1, dtype=np.int64)
    starts = train["start_pos"].to_numpy(dtype=np.int64)
    ends = train["end_pos"].to_numpy(dtype=np.int64) + 1
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    fit_mask = np.cumsum(diff[:-1]) > 0
    fit = features.loc[fit_mask]

    keep, dropped, coverage = [], {}, {}
    for base, columns in groups.items():
        series = fit[base]
        cov = float(series.notna().mean())
        coverage[base] = cov

        if is_calendar_col(base):
            # 日历列永远完全已知（恒不缺失），且装载侧禁止对日历列
            # 补零对齐（会把 0 误当成语义值），故不参与覆盖率/常数
            # 丢弃，始终保留（实践中日历列在训练期内均非常数）。
            keep.extend(columns)
            continue

        if cov < min_coverage or series.notna().sum() < min_obs:
            dropped[base] = "low_train_coverage_or_observations"
        elif series.dropna().nunique() <= 1:
            dropped[base] = "constant_in_train"
        else:
            keep.extend(columns)

    # 目标历史列必须保留，且置于最后。
    keep.append(TARGET)
    if len(keep) < 2:
        raise ValueError("没有足够的可用特征")

    stats = {}
    for col in keep:
        values = fit[col].dropna()
        if col == TARGET and len(values) < min_obs:
            raise ValueError("训练收益率有效值不足")
        if values.empty:
            raise ValueError(f"保留列 {col} 在训练集全空")

        if is_missing_col(col):
            # 二值标记保持 0/1。
            stats[col] = {"median": 1.0, "mean": 0.0, "scale": 1.0}
            continue

        if is_calendar_col(col):
            # 目标日日历列（时间标记列与旗标列）保持原值，不参与
            # 标准化（日历信息永远完全已知，取值本身已归一/为 0/1）。
            stats[col] = {"median": 0.0, "mean": 0.0, "scale": 1.0}
            continue

        median = float(values.median())
        filled = fit[col].fillna(median)
        mean = float(filled.mean())
        scale = float(filled.std(ddof=0))

        if not np.isfinite(scale) or scale < 1e-12:
            if col == TARGET:
                raise ValueError("训练收益率为常数")
            scale = 1.0

        stats[col] = {"median": median, "mean": mean, "scale": scale}

    return {
        "feature_columns": keep,
        "target": TARGET,
        "statistics": stats,
        "dropped_groups": dropped,
        "train_group_coverage": coverage,
        "fit_row_count": int(fit_mask.sum()),
        "fit_first_date": str(fit.index.min().date()),
        "fit_last_date": str(fit.index.max().date()),
    }




def transform(features, preprocessing):
    columns = preprocessing["feature_columns"]
    missing = set(columns) - set(features)
    if missing:
        raise ValueError(f"推理数据缺少训练特征定义：{sorted(missing)}")

    result = np.empty((len(features), len(columns)), dtype=np.float32)
    for i, col in enumerate(columns):
        p = preprocessing["statistics"][col]
        values = (
            (features[col].fillna(p["median"]) - p["mean"]) / p["scale"]
        )
        result[:, i] = values.to_numpy(dtype=np.float32)

    if not np.isfinite(result).all():
        raise ValueError("最终矩阵存在 NaN、无穷值或 float32 溢出")
    return result


