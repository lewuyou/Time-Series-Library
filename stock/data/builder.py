"""单股导出编排（prepare_one）与最新窗口/覆盖登记。"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock.data.calendar import calendar_fingerprint
from stock.data.common import SPLIT_LAYOUTS, VERSION, write_json
from stock.data.features import build_features
from stock.data.pit import history_drift_report, source_watermarks
from stock.data.preprocessing import fit_preprocessing, transform
from stock.data.schema import (
    TARGET,
    feature_schema_doc,
    is_companion_col,
    is_exogenous_col,
)
from stock.data.sources import load_prices, load_stock_aux
from stock.data.targets import target_spec
from stock.data.windows import make_samples
from stock.utils import sha256_file

json_write = write_json

def save_latest(out, matrix, features, core_valid, calendar, config):
    length = config["seq_len"]
    if len(matrix) < length or not core_valid.iloc[-length:].all():
        return {
            "saved": False,
            "reason": "latest_window_too_short_or_invalid",
        }

    input_date = features.index[-1]
    target_date = calendar[calendar.get_loc(input_date) + 1]
    np.savez_compressed(
        out / "latest.npz",
        X=matrix[-length:][None, ...],
        input_date=np.asarray([input_date], dtype="datetime64[D]"),
        target_date=np.asarray([target_date], dtype="datetime64[D]"),
    )
    return {
        "saved": True,
        "input_date": str(input_date.date()),
        "target_date": str(target_date.date()),
    }




def load_asof(args, con, calendar):
    if args.as_of:
        as_of = pd.Timestamp(args.as_of).normalize()
    else:
        value = con.execute(
            "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
        ).fetchone()[0]
        if value is None:
            raise ValueError("数据库没有后复权行情")
        as_of = pd.Timestamp(value).normalize()

    if as_of not in calendar:
        raise ValueError("as-of 不在交易日历内")
    if calendar.get_loc(as_of) + 1 >= len(calendar):
        raise ValueError("交易日历缺少下一交易日")
    return as_of




def coverage_report(features, valid_price, samples, train_end, val_end):
    index = features.index
    # train_test 布局没有验证区间：train <= train_end（即 fit_end），
    # test 直接从 train_end 之后开始。
    if val_end is None:
        masks = {
            "train": index <= train_end,
            "test": index > train_end,
        }
    else:
        masks = {
            "train": index <= train_end,
            "val": (index > train_end) & (index <= val_end),
            "test": index > val_end,
        }
    report = {}
    for name, mask in masks.items():
        if not mask.any():
            report[name] = {"rows": 0}
            continue
        part = features.loc[mask]
        in_split = samples["split"] == name
        report[name] = {
            "rows": int(mask.sum()),
            "valid_price_ratio": float(valid_price.loc[mask].mean()),
            # 可评分样本（label_valid）；可预测行数见 eligible_signals。
            "samples": int((in_split & samples["label_valid"]).sum()),
            "eligible_signals": int(in_split.sum()),
            "feature_nonmissing_ratio": {
                c: float(part[c].notna().mean()) for c in part
            },
        }
    return report




def prepare_one(con, row, args, calendar, as_of, macro_specs, config,
                calendar_info, frozen=None):
    symbol = row["symbol"]
    config = json.loads(json.dumps(config))
    config["baidu_keyword"] = config.get("baidu_keyword") or str(row["name"])

    # 保留整个日历覆盖区间用于预热。
    # 辅助来源额外向前读取，以便日历第一行仍能匹配最近可用观测。
    max_lookback = max(
        p["max_age_days"] + p["lag_days"] for p in config["policies"].values()
    )
    aux_start = calendar.min() - pd.Timedelta(days=max_lookback + 10)
    next_session = calendar[calendar.get_loc(as_of) + 1]

    prices = load_prices(con, symbol, calendar.min(), as_of)
    specs = load_stock_aux(
        con, symbol, config["baidu_keyword"], aux_start, next_session
    ) + macro_specs

    raw, features, audit, core_valid, valid_price, groups = build_features(
        prices, calendar, as_of, specs, config
    )

    if frozen is None:
        split_layout = getattr(args, "split_layout", None) or "train_val_test"
        if split_layout not in SPLIT_LAYOUTS:
            raise ValueError(f"不支持的 split_layout：{split_layout}")

        train_end = pd.Timestamp(args.train_end).normalize()
        if split_layout == "train_val_test":
            if args.val_end is None:
                raise ValueError("train_val_test 布局必须提供 val_end")
            val_end = pd.Timestamp(args.val_end).normalize()
        else:
            if args.val_end is not None:
                raise ValueError(
                    "train_test 布局没有验证集，不能提供 val_end"
                )
            val_end = None
        start = pd.Timestamp(args.start).normalize() if args.start else None

        samples, candidates, exclusions = make_samples(
            features, core_valid, config, train_end, val_end,
            start=start, tail=args.tail, split_layout=split_layout,
        )
        # 训练资格只由训练期（且标签有效）的样本量决定；验证/测试覆盖率
        # 是评估信息，不决定训练资格（否则未来评估期停牌/缺数/退市会
        # 反过来剔除训练完全合格的股票，形成测试期筛选偏差）。
        scoreable = samples[samples["label_valid"]]
        counts = scoreable["split"].value_counts()
        if int(counts.get("train", 0)) < args.min_train_samples:
            raise ValueError(
                f"train 样本不足：{int(counts.get('train', 0))} "
                f"< {args.min_train_samples}"
            )

        preprocessing = fit_preprocessing(
            features, samples, groups,
            args.min_feature_coverage, args.min_observations,
        )
        preprocessing.update({
            "version": VERSION,
            "symbol": symbol,
            "config": config,
            "split_layout": split_layout,
            "train_end": args.train_end,
            "val_end": args.val_end,
            # 样本选择口径登记：--start 限制最早目标日期（walk-forward
            # rolling 每折不同），--tail 限制最近候选目标日；两者都改变
            # 样本集合，导出复用与训练装载时都会比对，防止不同训练起点
            # 的导出被误复用。
            "sample_selection": {
                "start": str(start.date()) if start is not None else None,
                "tail": args.tail,
            },
            # validate 模式重拟合对比需要与训练一致的过滤阈值。
            "min_feature_coverage": args.min_feature_coverage,
            "min_observations": args.min_observations,
            # 日历指纹：infer 冻结日历参数并校验历史前缀。
            "calendar": calendar_fingerprint(
                calendar, as_of, calendar_info
            ),
            # 特征 schema 版本 + 稳定 hash（完整列描述另存
            # feature_schema.json，避免 252 只股票各存一份大 JSON）。
            "feature_schema": {
                key: value
                for key, value in feature_schema_doc(
                    preprocessing["feature_columns"]
                ).items()
                if key != "columns"
            },
            # 目标定义登记：预测标签不等同于可执行收益（后者见回测层）。
            "target_spec": target_spec(config["return_type"]),
        })

        report = coverage_report(
            features, valid_price, samples, train_end, val_end
        )
        # 验证/测试覆盖率登记（两种布局一致）：单股下限只用于报告对照，
        # 不决定训练资格。整体层面"该折是否有足够验证证据选模型"由
        # 训练装载检查（train_global_timexer）负责。
        report["eval_sample_minimum"] = {
            "required_per_stock": args.min_eval_samples,
            "enforced": False,
            "note": (
                "验证/测试覆盖率为评估信息，不决定训练资格；单股仅强制"
                "训练样本下限，评估期按实际有效样本参与"
            ),
            "per_stock_scoreable": {
                "train": int(counts.get("train", 0)),
                **(
                    {"val": int(counts.get("val", 0))}
                    if split_layout == "train_val_test" else {}
                ),
                "test": int(counts.get("test", 0)),
            },
        }
        report["candidate_target_dates"] = candidates
        # valid_samples 为可评分样本（label_valid）；eligible_samples 为
        # 可预测信号（含目标日无标签的行）。
        report["valid_samples"] = int(len(scoreable))
        report["eligible_samples"] = int(len(samples))
        report["valid_sample_ratio"] = (
            float(len(scoreable) / candidates) if candidates else 0.0
        )
        report["sample_exclusions"] = exclusions
        report["prediction_eligible_dates"] = (
            candidates
            - exclusions["insufficient_history"]
            - exclusions["invalid_input_window"]
        )
    else:
        preprocessing = frozen
        samples = None
        report = {
            "mode": "infer",
            "rows": len(features),
            f"latest_{config['seq_len']}_valid_core_rows": int(
                core_valid.iloc[-config["seq_len"]:].sum()
            ),
        }

    matrix = transform(features, preprocessing)
    # 旧导出目录（v8 及以前）没有登记 feature_schema/target_spec：infer
    # 时按当前列与配置补算，只用于写入本次 manifest，不回写冻结的
    # preprocessing（避免改变历史产物的契约）。
    schema_doc = feature_schema_doc(preprocessing["feature_columns"])
    schema_ref = preprocessing.get("feature_schema") or {
        key: value for key, value in schema_doc.items() if key != "columns"
    }
    target_reg = preprocessing.get("target_spec") or target_spec(
        config["return_type"]
    )

    warnings = []
    # 最新输入行的辅助特征缺失/过期（数据未到或超出 max_age）。
    aux_bases = [
        c for c in features.columns
        if is_exogenous_col(c) and not is_companion_col(c)
    ]
    stale_latest = [
        c for c in aux_bases if not np.isfinite(features[c].iloc[-1])
    ]
    if stale_latest:
        warnings.append(
            "最新输入行辅助特征缺失或过期（数据未到或超出 max_age）："
            + ", ".join(stale_latest[:10])
            + (" 等" if len(stale_latest) > 10 else "")
        )
    removed_days = calendar_info.get("removed_no_quote_days") or []
    if removed_days:
        warnings.append(
            "从日历剔除的无行情推导交易日："
            + ", ".join(d["date"] for d in removed_days)
            + "（如属采集故障而非休市，请改用权威 --calendar）"
        )
    partial_days = calendar_info.get("partial_coverage_days") or []
    if partial_days:
        warnings.append(
            f"存在 {len(partial_days)} 个行情覆盖不足的交易日，"
            "已保留在日历并标记为数据异常"
        )

    history_drift = None
    if frozen is not None:
        history_drift = history_drift_report(
            args.preprocessor, matrix, features,
            preprocessing["feature_columns"],
        )
        report["history_drift"] = history_drift
        if history_drift.get("checked") and history_drift["changed_rows"] > 0:
            msg = (
                f"历史特征行与训练期不一致：{history_drift['changed_rows']}/"
                f"{history_drift['compared_rows']} 行变化，max|Δ|="
                f"{history_drift['max_abs_diff']:.3g}（数据库历史数据可能被修订）"
            )
            warnings.append(msg)
            print(f"[WARN] {symbol} {msg}")
            if args.fail_on_history_drift:
                raise ValueError(msg + "；已按 --fail-on-history-drift 失败")

    if samples is not None:
        p = preprocessing["statistics"][TARGET]
        samples["y_scaled"] = (samples["y_raw"] - p["mean"]) / p["scale"]

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / symbol
    if final.exists():
        raise FileExistsError(
            f"输出目录已存在：{final}。请换一个新的 --out-dir，避免混用版本。"
        )

    temp = Path(tempfile.mkdtemp(prefix=f".{symbol}_", dir=root))
    try:
        raw.to_csv(temp / "aligned_values.csv", index_label="date")
        features.to_csv(
            temp / "features_unscaled.csv", index_label="date"
        )
        audit.to_csv(temp / "alignment_audit.csv", index_label="date")
        np.save(temp / "feature_matrix.npy", matrix)
        np.save(
            temp / "dates.npy",
            features.index.to_numpy(dtype="datetime64[D]"),
        )

        if samples is not None:
            samples.to_csv(temp / "samples.csv", index=False)
            # 训练前数据预览：逐交易日的标准化特征（与 feature_matrix.npy
            # 逐值一致，即模型实际吃到的数值）+ 当日原始收益率 + 按目标
            # 日期口径的切分标签（与 samples.csv 的 split 规则一致）。
            preview = pd.DataFrame(
                matrix, index=features.index,
                columns=preprocessing["feature_columns"],
            )
            preview["y_raw"] = features[TARGET].to_numpy(dtype=float)
            if split_layout == "train_test":
                preview["split"] = np.where(
                    features.index <= train_end, "train", "test"
                )
            else:
                preview["split"] = np.select(
                    [features.index <= train_end, features.index <= val_end],
                    ["train", "val"], default="test",
                )
            preview.to_csv(temp / "train_preview.csv", index_label="date")

        latest = save_latest(
            temp, matrix, features, core_valid, calendar, config
        )
        json_write(temp / "preprocessing.json", preprocessing)
        # 完整特征列描述（name/order/dtype/group + hash）：供审计与未来
        # 跨模型契约校验；preprocessing/manifest 只存版本与 hash 引用。
        json_write(
            temp / "feature_schema.json",
            schema_doc,
        )

        manifest = {
            "version": VERSION,
            "mode": args.mode,
            "created_at": datetime.now().astimezone().isoformat(),
            "symbol": symbol,
            "name": None if pd.isna(row["name"]) else str(row["name"]),
            "industry_snapshot": (
                None if pd.isna(row.get("industry_l2"))
                else str(row["industry_l2"])
            ),
            "as_of": str(as_of.date()),
            # 预测时点：下一交易日 decision_hour:minute；导出时点未必等于
            # 该时刻（例如周五收盘后导出的是"周一 08:30"的窗口，周末及
            # 周一早上的信息尚未到齐），由 source_watermarks 反映数据水位。
            "forecast_time": (
                next_session
                + pd.Timedelta(
                    hours=config["decision_hour"],
                    minutes=config["decision_minute"],
                )
            ).isoformat(),
            "source_watermarks": source_watermarks(
                con, symbol, config["baidu_keyword"]
            ),
            "warnings": warnings,
            "seq_len": config["seq_len"],
            "pred_len": 1,
            "return_type": config["return_type"],
            "target": TARGET,
            "target_spec": target_reg,
            "feature_schema": schema_ref,
            "enc_in": matrix.shape[1],
            "matrix_shape": list(matrix.shape),
            "latest": latest,
            "quality": report,
            "config": config,
            "db_path": str(Path(args.db).resolve()),
            "calendar": calendar_info,
            "macro_mapping": [
                {k: v for k, v in spec.items() if k != "series"}
                for spec in macro_specs
            ],
            "limitations": [
                "原表没有实际发布时间，采用配置的可用时间假设；"
                "source_watermarks 只是入库水位，不能证明 point-in-time。",
                "历史复权、财务和宏观数据可能存在事后修订；"
                "infer 的 history_drift 只能发现、不能消除该问题。",
                "默认当前股票简称匹配百度，未自动恢复历史更名。",
                "当前行业、市值和存续名单只用于选股，存在样本选择局限。",
                "交易日历只覆盖 --calendar-start 之后的历史。",
                "数据库推导的日历以行情数据校准，未来交易日来自"
                "holiday_calendar，可能存在误差。",
                "未自动识别接口以平价零量记录表示的停牌，需要来源核验。",
                "预测的是统计口径收益率，不等于可实现交易收益"
                "（t+1 早 08:30 出信号后无法按 t 收盘价成交）。",
            ],
        }
        if frozen is not None:
            manifest["frozen_preprocessing_sha256"] = sha256_file(
                args.preprocessor
            )
        manifest["artifact_sha256"] = {
            p.name: sha256_file(p) for p in sorted(temp.iterdir())
        }
        json_write(temp / "manifest.json", manifest)
        os.replace(temp, final)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    count_text = ""
    if samples is not None:
        scoreable_counts = samples.loc[
            samples["label_valid"], "split"
        ].value_counts().to_dict()
        unscoreable = int((~samples["label_valid"]).sum())
        count_text = (
            str(scoreable_counts)
            + (f" | unscoreable={unscoreable}" if unscoreable else "")
        )
    print(
        f"[OK] {symbol} {row['name']} | "
        f"F={matrix.shape[1]} | {count_text} | latest={latest['saved']}"
        + (f" | warnings={len(warnings)}" if warnings else "")
    )
    return {"symbol": symbol, "status": "ok", "directory": str(final)}


