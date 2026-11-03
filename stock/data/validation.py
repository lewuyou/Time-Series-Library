"""导出目录离线验收（--mode validate 的全部检查项）。"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stock.data.preprocessing import fit_preprocessing, transform
from stock.data.schema import TARGET

def _stats_close(a, b, atol=1e-9):
    """比较两份 preprocessing 统计量（容差比较，避免浮点误报）。"""
    if set(a) != set(b):
        return False, f"列集合不同：{sorted(set(a) ^ set(b))[:8]}"
    for col in a:
        x, y = a[col], b[col]
        if set(x) != set(y):
            return False, f"{col} 统计字段不同"
        for k in x:
            if not np.isclose(float(x[k]), float(y[k]), rtol=0, atol=atol):
                return False, f"{col}.{k}: {x[k]} != {y[k]}"
    return True, ""




def validate_dataset(data_dir):
    """导出目录的离线验收测试（不连数据库）。

    覆盖审查建议 12.1 / 12.2（结构性子集）/ 12.3 / 12.6：
      1. 文件齐全（训练导出含 train_preview.csv）、矩阵有限、形状与列
         定义一致、日期有序唯一；
      2. 样本索引与标签一致（target_pos == end_pos + 1、窗口长度、
         日期与位置一致、目标晚于输入）；train_preview.csv 与矩阵逐值
         一致；
      3. y_raw 可由相邻后复权收盘价重算，y_scaled 与矩阵末列一致；
      4. 切分边界：train/val/test 的目标日期互不越界；
      5. 对数收益率 expm1 与简单收益率一致；
      6. 审计 available <= decision（未来数据不可用断言的复核）；
      7. 验证/测试期数值扰动后重拟合，统计量与特征列不变；
      8. 未来行扰动后重 transform，历史行矩阵不变；
      9. latest.npz 形状与有限性。
    返回是否全部通过；任何失败项都会打印。
    """
    root = Path(data_dir)
    checks = []

    def record(name, ok, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if ok else "fail",
            "detail": str(detail),
        })
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {name}"
        if detail:
            line += f"：{detail}"
        print(line)
        return ok

    # 必需文件由 manifest 登记的模式决定，而不是"缺 samples.csv 就当
    # infer 跳过关键检查"——训练导出意外丢失样本文件必须判为验收失败。
    required = [
        "feature_matrix.npy", "dates.npy", "preprocessing.json",
        "aligned_values.csv", "alignment_audit.csv", "manifest.json",
        "features_unscaled.csv",
    ]
    missing = [f for f in required if not (root / f).exists()]
    record("required_files", not missing, f"缺失 {missing}" if missing else "")
    if missing:
        return False

    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    export_mode = manifest.get("mode")
    if export_mode not in ("prepare", "infer"):
        record(
            "manifest_mode", False,
            f"manifest.mode={export_mode!r} 无法识别，无法判定必需文件",
        )
        return False

    has_samples = (root / "samples.csv").exists()
    if export_mode == "prepare" and not has_samples:
        record(
            "samples_required_for_prepare", False,
            "训练导出（prepare）缺失 samples.csv，不能按 infer 跳过检查",
        )
        return False
    if export_mode == "prepare":
        required.append("train_preview.csv")
        missing = [f for f in required if not (root / f).exists()]
        record(
            "required_files_prepare", not missing,
            f"缺失 {missing}" if missing else "",
        )
        if missing:
            return False

    matrix = np.load(root / "feature_matrix.npy")
    dates = pd.DatetimeIndex(np.load(root / "dates.npy"))
    prep = json.loads(
        (root / "preprocessing.json").read_text(encoding="utf-8")
    )
    raw = pd.read_csv(
        root / "aligned_values.csv", index_col=0, parse_dates=True
    )
    columns = prep["feature_columns"]
    return_type = prep["config"].get("return_type", "simple")
    seq_len = int(prep["config"]["seq_len"])

    # infer 导出不生成 samples.csv / 训练统计复核，样本类检查跳过。
    samples = (
        pd.read_csv(root / "samples.csv") if has_samples else None
    )
    if samples is None:
        record(
            "sample_checks_skipped", True,
            "无 samples.csv（infer 导出），跳过样本与切分检查",
        )
    elif "label_valid" not in samples.columns:
        record(
            "label_valid_column", False,
            "samples.csv 缺少 label_valid 列（旧版导出），请用当前版本"
            "重新导出后再验收",
        )
        samples = None

    record("matrix_finite", bool(np.isfinite(matrix).all()))
    record(
        "matrix_shape",
        matrix.ndim == 2
        and matrix.shape[0] == len(dates)
        and matrix.shape[1] == len(columns),
        f"{matrix.shape} vs ({len(dates)}, {len(columns)})",
    )
    record(
        "dates_sorted_unique",
        bool(dates.is_monotonic_increasing and dates.is_unique),
    )
    record("target_last_column", columns[-1] == TARGET)

    train_end = pd.Timestamp(prep["train_end"])
    split_layout = prep.get("split_layout", "train_val_test")
    val_end = (
        pd.Timestamp(prep["val_end"])
        if prep.get("val_end") is not None else None
    )
    if split_layout == "train_test" and val_end is not None:
        raise ValueError("train_test 布局的 preprocessing 不应包含 val_end")

    if samples is not None:
        sp = samples["start_pos"].to_numpy(dtype=int)
        ep = samples["end_pos"].to_numpy(dtype=int)
        tp = samples["target_pos"].to_numpy(dtype=int)
        # 先验证位置边界再索引，坏数据输出验收失败项而不是 IndexError。
        in_range = bool(
            (sp >= 0).all()
            and (ep < len(dates)).all()
            and (tp < len(dates)).all()
        )
        record(
            "sample_positions_in_range", in_range,
            f"max_pos={max(tp.max(), ep.max()) if len(tp) else -1}, "
            f"dates={len(dates)}",
        )

        allowed_splits = (
            {"train", "test"} if split_layout == "train_test"
            else {"train", "val", "test"}
        )
        record(
            "split_values_allowed",
            set(samples["split"].dropna().unique()) <= allowed_splits,
            f"observed={sorted(samples['split'].dropna().unique())}",
        )

        label_valid = samples["label_valid"].to_numpy(dtype=bool)
        record(
            "label_invalid_rows_nan_y_raw",
            bool(samples.loc[~label_valid, "y_raw"].isna().all()),
            f"unscoreable={int((~label_valid).sum())}",
        )

        if in_range:
            record(
                "target_pos_follows_window", bool((tp == ep + 1).all())
            )
            record(
                "window_length", bool((ep - sp + 1 == seq_len).all()),
                f"seq_len={seq_len}",
            )
            input_dates = pd.to_datetime(samples["input_date"]).to_numpy()
            target_dates = pd.to_datetime(samples["target_date"]).to_numpy()
            record(
                "sample_pos_date_consistency",
                bool(
                    (dates[ep] == input_dates).all()
                    and (dates[tp] == target_dates).all()
                ),
            )
            record(
                "target_after_input", bool((dates[tp] > dates[ep]).all())
            )

            close = raw["px_close"].to_numpy(dtype=float)
            if return_type == "log":
                y_ref = np.log(close[tp] / close[tp - 1])
            else:
                y_ref = close[tp] / close[tp - 1] - 1.0
            y_raw = samples["y_raw"].to_numpy(dtype=float)
            err = (
                float(np.max(np.abs(y_ref[label_valid] - y_raw[label_valid])))
                if label_valid.any() else 0.0
            )
            record(
                "y_raw_recompute_from_close", err <= 1e-9,
                f"max|Δ|={err:.2e}",
            )

            p = prep["statistics"][TARGET]
            y_scaled_ref = (y_raw - p["mean"]) / p["scale"]
            err2 = (
                float(np.max(np.abs(
                    matrix[tp[label_valid], -1] - y_scaled_ref[label_valid]
                )))
                if label_valid.any() else 0.0
            )
            record(
                "y_scaled_matches_matrix", err2 <= 1e-4,
                f"max|Δ|={err2:.2e}",
            )

            if return_type == "log":
                simple_ref = close[tp] / close[tp - 1] - 1.0
                err3 = (
                    float(np.max(np.abs(
                        np.expm1(y_raw[label_valid])
                        - simple_ref[label_valid]
                    )))
                    if label_valid.any() else 0.0
                )
                record(
                    "log_to_simple_consistency", err3 <= 1e-12,
                    f"max|Δ|={err3:.2e}",
                )

            tdate = pd.to_datetime(samples["target_date"])
            split = samples["split"]
            if split_layout == "train_test":
                ok = (
                    (tdate[split == "train"] <= train_end).all()
                    and (tdate[split == "test"] > train_end).all()
                    and (split == "val").sum() == 0
                )
            else:
                ok = (
                    (tdate[split == "train"] <= train_end).all()
                    and (tdate[split == "val"] > train_end).all()
                    and (tdate[split == "val"] <= val_end).all()
                    and (tdate[split == "test"] > val_end).all()
                )
            record("split_boundaries", bool(ok))

            # rolling 导出（--start）时训练目标日不得早于登记的训练起点。
            start_value = (prep.get("sample_selection") or {}).get("start")
            if start_value is not None:
                record(
                    "train_start_lower_bound",
                    bool(
                        (tdate[split == "train"] >= pd.Timestamp(start_value)).all()
                    ),
                    f"start={start_value}",
                )

        # 训练前数据预览 CSV 与矩阵逐值一致（导出即训练输入的复核）。
        # CSV 十进制往返存在浮点表示噪声（float32 最短表示 ~1e-6 相对
        # 误差），用相对容差比较；真实数据错误远大于该量级。
        preview = pd.read_csv(
            root / "train_preview.csv", index_col=0, parse_dates=True
        )
        err = float(
            np.nanmax(np.abs(
                preview[columns].to_numpy(dtype=float)
                - matrix.astype(np.float64)
            )) if len(preview) == len(dates) else np.inf
        )
        preview_ok = (
            list(preview.columns) == columns + ["y_raw", "split"]
            and len(preview) == len(dates)
            and np.allclose(
                preview[columns].to_numpy(dtype=float),
                matrix.astype(np.float64), rtol=1e-6, atol=1e-9,
            )
        )
        record(
            "train_preview_matches_matrix", bool(preview_ok),
            f"shape={preview.shape} vs {matrix.shape}, max|Δ|={err:.2e}",
        )

    audit = pd.read_csv(root / "alignment_audit.csv", index_col=0)
    decision = pd.to_datetime(audit["decision_time"])
    bad = 0
    for col in audit.columns:
        if col.endswith("__available"):
            available = pd.to_datetime(audit[col])
            bad += int((available.notna() & (available > decision)).sum())
    record("audit_available_le_decision", bad == 0, f"违规 {bad} 行")

    features = pd.read_csv(
        root / "features_unscaled.csv", index_col=0, parse_dates=True
    )
    perturbed = features.copy()
    eval_mask = perturbed.index > train_end
    perturbed.loc[eval_mask, columns] = (
        perturbed.loc[eval_mask, columns] * 1.5 + 0.25
    )
    if samples is not None:
        groups = {}
        for col in columns:
            if col == TARGET:
                continue  # TARGET 由 fit_preprocessing 固定追加在末尾
            base = re.sub(r"__(missing|age_days)$", "", col)
            groups.setdefault(base, []).append(col)
        try:
            refit = fit_preprocessing(
                perturbed, samples, groups,
                float(prep.get("min_feature_coverage", 0.60)),
                int(prep.get("min_observations", 100)),
            )
            same_columns = refit["feature_columns"] == columns
            same_stats, why = _stats_close(
                refit["statistics"], prep["statistics"]
            )
            record(
                "preprocessing_immune_to_eval_data",
                bool(same_columns and same_stats),
                "" if same_columns and same_stats else (why or "特征列不同"),
            )
        except Exception as exc:
            record("preprocessing_immune_to_eval_data", False, str(exc))

    original = transform(features, prep)
    after = transform(perturbed, prep)
    history_mask = np.asarray(features.index <= train_end)
    record(
        "future_perturbation_invariance",
        bool(np.array_equal(original[history_mask], after[history_mask])),
    )

    gaps = dates.to_series().diff().dt.days.dropna()
    record(
        "calendar_gap_info", True,
        f"最大相邻交易日间隔 {int(gaps.max())} 天（软信息，不判失败）",
    )

    if (root / "latest.npz").exists():
        latest = np.load(root / "latest.npz")
        ok = (
            latest["X"].shape == (1, seq_len, len(columns))
            and bool(np.isfinite(latest["X"]).all())
        )
        record(
            "latest_window", bool(ok),
            f"shape={latest['X'].shape}, input_date={latest['input_date']}",
        )

    passed = sum(1 for c in checks if c["status"] == "pass")
    all_ok = passed == len(checks)
    print(f"validate {root}: {passed}/{len(checks)} 项通过" + (
        "" if all_ok else "，存在失败项"
    ))
    return all_ok


