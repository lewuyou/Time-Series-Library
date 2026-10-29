# -*- coding: utf-8 -*-
"""构造带已知"多重变换"数据生成过程（DGP）的合成测试数据集。

目的：
    用一个真值完全已知的数据集验证 train_global_timexer.py 中 TimeXer
    的有效性——目标值由输入窗口内的特征经过多重变换（滚动均值、
    非线性压缩、跨变量交互、周期变换、滞后加权、慢变量门控）生成，
    外加少量不可约噪声。若模型与流程正确，测试集上 y_pred 的 MSE 应
    显著低于 zero / train-mean / persistence 三个基准，且接近噪声地板。

DGP（全部因果，只用窗口内信息；t 为目标日前一交易日）：
    r1_t = mean(syn_momentum[t-4..t])            # 变换1：5 日滚动均值
    r2_t = mean(baidu_search_all[t-9..t])        # 变换2：10 日滚动均值
    n1_t = tanh(2.5 * rzrq_fin_tval_ratio[t])    # 变换3：非线性压缩
    n2_t = r1_t * n1_t                           # 变换4：跨变量交互
    l1_t = 0.5*syn_volatility[t] + 0.3*[t-1]
           + 0.2*[t-2]                           # 变换5：滞后加权(EMA3)
    y_{t+1} = 0.7*tanh(3*n2_t)                   # 变换6：再非线性
            + 0.4*sin(3*r2_t)                    # 变换7：周期变换
            + 0.3*l1_t
            + 0.2*tanh(macro_close_SYNSRC_A[t])  # 变换8：慢变量门控
            + eps_{t+1},  eps ~ N(0, noise_std)

设计要点：
    - 各股票共享同一函数形式、不同的随机实现（模拟全局共享模型）；
    - y 对自身历史近似无自相关（不依赖 stock_return 历史），
      persistence 基准应失效，有效性只能来自外生特征注意力；
    - px_* 价格类基数列（px_volume/px_turnover）与 macro_B 为
      纯干扰特征，检验模型抗噪能力；
    - 输出目录结构与 export_global.py 兼容，直接供
      train_global_timexer.py --mode train 使用。

输出（--out-dir）：
    export_config.json     导出任务登记（train_end/val_end/seq_len/...）
    export_results.csv     symbol,status,error（全部 ok）
    <symbol>/preprocessing.json / feature_matrix.npy / dates.npy /
        samples.csv        与真实导出一致的逐股产物

用法：
    python make_synthetic_dataset.py --out-dir dataset/synthetic_001 \
        --symbols 8 --days 960 --seq-len 60 --seed 42 --noise-ratio 0.15

    python train_global_timexer.py --mode train \
        --data-root dataset/synthetic_001 \
        --out-dir checkpoints/synthetic_timexer_001 \
        --epochs 30 --patience 5 --patch-len 10 --device auto
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "stock_return"
DROP_BASES = {"px_open", "px_close", "px_high", "px_low"}
WARMUP = 10  # r2 需要 10 日窗口；target 日 < WARMUP 的样本 label_valid=False

# (基数列, 是否附带 __age_days 列)。price 组是 px_* 干扰列。
BASE_SPEC = [
    ("px_open", False),
    ("px_close", False),
    ("px_high", False),
    ("px_low", False),
    ("px_volume", False),
    ("px_turnover", False),
    ("syn_momentum", False),
    ("syn_volatility", False),
    ("baidu_search_all", True),
    ("rzrq_fin_tval_ratio", True),
    ("macro_close_SYNSRC_A", True),
    ("macro_close_SYNSRC_B", True),
]

DGP_COEFFS = {
    "interaction_gain": 0.7,   # 0.7 * tanh(3 * r1 * tanh(2.5 * rzrq))
    "interaction_slope": 3.0,
    "rzrq_compress": 2.5,
    "periodic_gain": 0.4,      # 0.4 * sin(3 * r2)
    "periodic_slope": 3.0,
    "vol_ema_gain": 0.3,       # 0.3 * (0.5v + 0.3v1 + 0.2v2)
    "macro_gate_gain": 0.2,    # 0.2 * tanh(macro_A)
}

# 任务拆分（实验4）：单分量数据集，用于逐项检验可学性。
# mom_mean  纯滚动均值       y = 0.7 * r1
# cross     纯跨变量乘积     y = 0.7 * (r1 * tanh(2.5*rzrq))
# interaction 嵌套tanh交互   y = 0.7 * tanh(3 * r1 * tanh(2.5*rzrq))
# periodic  纯周期变换       y = 0.4 * sin(3 * r2)
# vol_ema   纯滞后加权       y = 0.3 * l1
# macro_gate 纯慢变量门控    y = 0.2 * tanh(macro_A)
# all       全部分量组合（默认）
TASK_CHOICES = [
    "all", "mom_mean", "cross", "interaction",
    "periodic", "vol_ema", "macro_gate",
]

TASK_FORMULAS = {
    "all": (
        "y = 0.7*tanh(3*mean5(mom)*tanh(2.5*rzrq)) + 0.4*sin(3*mean10(baidu))"
        " + 0.3*(0.5v+0.3v1+0.2v2) + 0.2*tanh(macroA) + eps"
    ),
    "mom_mean": "y = 0.7*mean5(mom) + eps",
    "cross": "y = 0.7*(mean5(mom)*tanh(2.5*rzrq)) + eps",
    "interaction": "y = 0.7*tanh(3*mean5(mom)*tanh(2.5*rzrq)) + eps",
    "periodic": "y = 0.4*sin(3*mean10(baidu)) + eps",
    "vol_ema": "y = 0.3*(0.5v+0.3v1+0.2v2) + eps",
    "macro_gate": "y = 0.2*tanh(macroA) + eps",
}


def rolling_mean(x, window):
    return pd.Series(x).rolling(window, min_periods=window).mean().to_numpy()


def build_raw_features(rng, n_days):
    """原始（未标准化）特征矩阵 dict：基数列 -> 一维数组。"""
    def ar1(phi, sigma=1.0):
        innovations = rng.normal(0.0, sigma, n_days)
        out = np.empty(n_days)
        out[0] = innovations[0]
        for t in range(1, n_days):
            out[t] = phi * out[t - 1] + innovations[t]
        return out

    raw = {
        # 干扰列：随机游走价格/量（标准化后进入模型，但与 y 无关）
        "px_open": 100.0 + np.cumsum(rng.normal(0, 0.5, n_days)),
        "px_close": 100.0 + np.cumsum(rng.normal(0, 0.5, n_days)),
        "px_high": 101.0 + np.cumsum(rng.normal(0, 0.5, n_days)),
        "px_low": 99.0 + np.cumsum(rng.normal(0, 0.5, n_days)),
        "px_volume": np.cumsum(np.abs(rng.normal(0, 1.0, n_days))) + 10.0,
        "px_turnover": np.abs(rng.normal(0.005, 0.002, n_days)),
        # 信息列
        "syn_momentum": ar1(0.6),
        "syn_volatility": 1.2 * np.abs(ar1(0.3)),
        "baidu_search_all": ar1(0.8),
        "rzrq_fin_tval_ratio": ar1(0.5),
        "macro_close_SYNSRC_A": ar1(0.9),   # 慢变量（门控项来源）
        "macro_close_SYNSRC_B": ar1(0.7),   # 干扰慢变量
    }
    return raw


def dgp_signal(raw, c, task="all"):
    """多重变换信号：只用截至 t 的信息，返回 signal[t]（t 处已可计算）。

    task 用于任务拆分实验：只保留对应分量（见 TASK_FORMULAS）。"""
    r1 = rolling_mean(raw["syn_momentum"], 5)
    r2 = rolling_mean(raw["baidu_search_all"], 10)
    n1 = np.tanh(c["rzrq_compress"] * raw["rzrq_fin_tval_ratio"])
    n2 = r1 * n1
    vol = raw["syn_volatility"]
    l1 = np.full_like(vol, np.nan)
    l1[2:] = 0.5 * vol[2:] + 0.3 * vol[1:-1] + 0.2 * vol[:-2]
    gate = np.tanh(raw["macro_close_SYNSRC_A"])
    parts = {
        "interaction": c["interaction_gain"]
        * np.tanh(c["interaction_slope"] * n2),
        "periodic": c["periodic_gain"]
        * np.sin(c["periodic_slope"] * r2),
        "vol_ema": c["vol_ema_gain"] * l1,
        "macro_gate": c["macro_gate_gain"] * gate,
    }
    if task == "all":
        return sum(parts.values())
    if task == "mom_mean":
        return c["interaction_gain"] * r1
    if task == "cross":
        return c["interaction_gain"] * n2
    return parts[task]


def column_layout(base_spec):
    """按真实导出顺序展开：值列、[__age_days]、__missing。"""
    columns = []
    for base, has_age in base_spec:
        columns.append(base)
        if has_age:
            columns.append(base + "__age_days")
        columns.append(base + "__missing")
    columns.append(TARGET)
    return columns


# --engineered：额外注入的因果构造特征（实验3，TimeXer+显式特征）。
# 只给中间量，不给最终非线性（sin/tanh 仍需模型自己学）。
ENGINEERED_BASES = [
    ("syn_feat_mom5", False),     # r1 = mean5(mom)
    ("syn_feat_cross", False),    # n2 = r1 * tanh(2.5*rzrq)
    ("syn_feat_heat10", False),   # r2 = mean10(baidu)
    ("syn_feat_volema", False),   # l1 = 0.5v+0.3v1+0.2v2
]


def engineered_values(raw):
    r1 = rolling_mean(raw["syn_momentum"], 5)
    r2 = rolling_mean(raw["baidu_search_all"], 10)
    n2 = r1 * np.tanh(2.5 * raw["rzrq_fin_tval_ratio"])
    vol = raw["syn_volatility"]
    l1 = np.full_like(vol, np.nan)
    l1[2:] = 0.5 * vol[2:] + 0.3 * vol[1:-1] + 0.2 * vol[:-2]
    return {
        "syn_feat_mom5": r1,
        "syn_feat_cross": n2,
        "syn_feat_heat10": r2,
        "syn_feat_volema": l1,
    }


def robust_stats(values):
    """训练期 robust 标准化参数（与真实导出一致的 median/mean/scale 口径）。

    nan 感知：工程特征列头部因滚动窗口未满为 NaN，统计时剔除；
    标准化后这些位置另行填 0（=均值水平）。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        values = np.zeros(1)
    scale = float(
        np.percentile(values, 75) - np.percentile(values, 25)
    )
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return {
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "scale": scale,
    }


def make_symbol(rng, symbol, dates, seq_len, train_end_idx, val_end_idx,
                noise_std, keyword, task="all", engineered=False):
    n_days = len(dates)
    raw = build_raw_features(rng, n_days)
    signal = dgp_signal(raw, DGP_COEFFS, task)
    base_spec = list(BASE_SPEC) + (ENGINEERED_BASES if engineered else [])

    # y[t]（t 日目标值）= 截至 t-1 的信号 + 当日标签噪声；
    # 前 WARMUP 日因滚动窗口未满不可计算，填 0 并在样本表置 label_valid=False。
    y = np.zeros(n_days, dtype=np.float64)
    noise = rng.normal(0.0, noise_std, n_days)
    y[WARMUP:] = signal[WARMUP - 1:-1] + noise[WARMUP:]

    # ---- 组装原始矩阵（值列 + 常量标记列），再做训练期标准化 ----
    columns = column_layout(base_spec)
    fit_rows = slice(0, train_end_idx + 1)  # 标准化只用训练期
    eng = engineered_values(raw) if engineered else {}
    matrix_raw = {}
    for base, has_age in base_spec:
        if base in eng:              # .get 的 fallback 是急切求值，须显式分支
            matrix_raw[base] = eng[base]
        else:
            matrix_raw[base] = raw[base]
        if has_age:
            matrix_raw[base + "__age_days"] = np.full(n_days, 2.0)
        matrix_raw[base + "__missing"] = np.zeros(n_days)
    matrix_raw[TARGET] = y

    statistics = {}
    standardized = np.empty((n_days, len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        stats = robust_stats(matrix_raw[col][fit_rows])
        statistics[col] = stats
        col_std = (matrix_raw[col] - stats["mean"]) / stats["scale"]
        # 工程特征头部 NaN（滚动窗口未满）填 0 = 均值水平，保证窗口有限。
        col_std = np.where(np.isfinite(col_std), col_std, 0.0)
        standardized[:, j] = col_std.astype(np.float32)

    # ---- 样本表：target_pos = end_pos + 1，按目标日划分 split ----
    target_stats = statistics[TARGET]
    rows = []
    for end in range(seq_len - 1, n_days - 1):
        target = end + 1
        target_date = str(dates[target])
        if target_date <= str(dates[train_end_idx]):
            split = "train"
        elif target_date <= str(dates[val_end_idx]):
            split = "val"
        else:
            split = "test"
        y_raw = float(y[target])
        rows.append({
            "start_pos": end - seq_len + 1,
            "end_pos": end,
            "target_pos": target,
            "input_date": str(dates[end]),
            "target_date": target_date,
            "split": split,
            "label_valid": target >= WARMUP,
            "y_raw": y_raw,
            "y_scaled": (y_raw - target_stats["mean"]) / target_stats["scale"],
        })
    samples = pd.DataFrame(rows)

    config = {
        "seq_len": seq_len,
        "return_type": "simple",
        "macro_fields": ["close"],
        "macro_symbols": ["SYNSRC:A", "SYNSRC:B"],
        "decision_hour": 8,
        "decision_minute": 30,
        "baidu_keyword": keyword,
    }
    preprocessing = {
        "feature_columns": columns,
        "target": TARGET,
        "statistics": statistics,
        "dropped_groups": {},
        "train_group_coverage": {
            base: 1.0 for base, _ in base_spec
        },
        "fit_row_count": train_end_idx + 1,
        "fit_first_date": str(dates[0]),
        "fit_last_date": str(dates[train_end_idx]),
        "version": "synthetic_timexer_v1",
        "symbol": symbol,
        "config": config,
        "train_end": str(dates[train_end_idx]),
        "val_end": str(dates[val_end_idx]),
        "sample_selection": {"start": None, "tail": None},
        "split_layout": "train_val_test",
        "calendar": {
            "source": "synthetic",
            "path": None,
            "sha256_of_calendar_csv": None,
            "trading_days_up_to_as_of": n_days,
            "sha256": None,
        },
    }

    signal_std = float(np.nanstd(signal[WARMUP:]))
    return {
        "symbol": symbol,
        "preprocessing": preprocessing,
        "matrix": standardized,
        "dates": dates,
        "samples": samples,
        "config": config,
        "signal_std": signal_std,
        "noise_std": noise_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="生成多重变换合成数据集（train_global_timexer 兼容格式）"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--symbols", type=int, default=8)
    parser.add_argument("--days", type=int, default=960)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-ratio", type=float, default=0.15,
        help="标签噪声标准差 / 信号标准差",
    )
    parser.add_argument(
        "--task", choices=TASK_CHOICES, default="all",
        help="任务拆分：只用对应分量生成 y（实验4）",
    )
    parser.add_argument(
        "--engineered", action="store_true",
        help="额外注入因果构造特征列（实验3：TimeXer+显式特征）",
    )
    parser.add_argument("--start-date", default="2023-01-02")
    args = parser.parse_args()

    if args.days < 200:
        raise ValueError("days 过短，训练样本不足")

    rng = np.random.default_rng(args.seed)
    dates = np.array(
        pd.bdate_range(args.start_date, periods=args.days).values,
        dtype="datetime64[D]",
    )

    train_end_idx = int(args.days * 0.70)
    val_end_idx = int(args.days * 0.85)

    # 先用 probe 批估计信号尺度，确定绝对噪声强度（跨股票一致）。
    probe = make_symbol(
        np.random.default_rng(args.seed + 999_001), "PROBE", dates,
        args.seq_len, train_end_idx, val_end_idx, 0.0, "probe",
        task=args.task, engineered=args.engineered,
    )
    noise_std = args.noise_ratio * probe["signal_std"]

    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} 非空，请换新目录")
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.symbols):
        symbol = f"SYN{i + 1:03d}"
        record = make_symbol(
            np.random.default_rng(args.seed + 1_000_000 + i), symbol, dates,
            args.seq_len, train_end_idx, val_end_idx, noise_std,
            f"合成股票{i + 1:03d}",
            task=args.task, engineered=args.engineered,
        )
        directory = out / symbol
        directory.mkdir(parents=True)

        np.save(directory / "feature_matrix.npy", record["matrix"])
        np.save(directory / "dates.npy", record["dates"])
        record["samples"].to_csv(directory / "samples.csv", index=False)
        (directory / "preprocessing.json").write_text(
            json.dumps(
                record["preprocessing"], ensure_ascii=False, indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        results.append({"symbol": symbol, "status": "ok", "error": ""})

        counts = (
            record["samples"]
            .query("label_valid")
            .groupby("split")
            .size()
            .to_dict()
        )
        print(
            f"{symbol}: signal_std={record['signal_std']:.4f} "
            f"noise_std={noise_std:.4f} samples={counts}"
        )

    pd.DataFrame(results).to_csv(out / "export_results.csv", index=False)

    export_config = {
        "export_version": "synthetic_export_v1",
        "db": None,
        "as_of": str(dates[-1]),
        "train_start": None,
        "train_end": str(dates[train_end_idx]),
        "val_end": str(dates[val_end_idx]),
        "test_start": str(dates[val_end_idx + 1]),
        "seq_len": args.seq_len,
        "split_layout": "train_val_test",
        "symbols": [r["symbol"] for r in results],
        "dataset_version": "synthetic_timexer_v1",
        "return_type": "simple",
        "decision_hour": 8,
        "decision_minute": 30,
        "dgp": {
            "task": args.task,
            "engineered": args.engineered,
            "formula": TASK_FORMULAS[args.task],
            "signal_std": probe["signal_std"],
            "noise_std": noise_std,
            "noise_ratio": args.noise_ratio,
            "theoretical_max_r2": (
                probe["signal_std"] ** 2
                / (probe["signal_std"] ** 2 + noise_std ** 2)
            ),
            "seed": args.seed,
        },
    }
    (out / "export_config.json").write_text(
        json.dumps(export_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"完成：symbols={args.symbols}, days={args.days}, "
        f"train_end={export_config['train_end']}, "
        f"val_end={export_config['val_end']}, "
        f"noise_std={noise_std:.4f}, "
        f"理论 R2 上限={export_config['dgp']['theoretical_max_r2']:.3f}"
    )
    print(f"输出目录：{out}")


if __name__ == "__main__":
    main()
