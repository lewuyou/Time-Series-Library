"""推理器：单股最新窗口预测 + 全池批量打分。

predict 自 train_global_timexer.py 原样迁出；score 自 predict_daily.py
原样迁出。两处仅把模型构造/前向改为经适配器（按 checkpoint 的
format 字段反查），数值行为与原脚本逐位一致。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stock.adapters import adapter_for_checkpoint
from stock.data.schema import (
    TARGET,
    align_features,
    compute_ex_mask,
    exogenous_mask_layout,
    feature_schema_doc,
    feature_schema_hash,
    is_calendar_col,
    is_mark_col,
)
from stock.inference.contracts import PredictionStatus
from stock.utils import read_json, sha256_file


def calendar_split(x_full, schema, config):
    """按 checkpoint 的 calendar_mode 拆分 x_enc / x_mark。

    推理端严格复现训练协议（登记在 model_config），不接受命令行覆盖；
    旧 checkpoint 缺 calendar_mode 按 columns（历史协议）处理。
    x_full：已对齐到 schema 的完整窗口 [..., seq_len, len(schema)]。
    返回 (x, x_mark)；x_mark=None 表示无标记通道（columns/none）。
    """
    mode = config.get("calendar_mode", "columns")
    if mode not in ("time_marks", "columns", "none"):
        raise ValueError(f"未知 calendar_mode：{mode}")
    if mode == "columns":
        return x_full, None
    if mode == "time_marks":
        mark_pos = [i for i, c in enumerate(schema[:-1]) if is_mark_col(c)]
        mark_set = set(mark_pos)
        keep = [i for i in range(len(schema)) if i not in mark_set]
        x_mark = x_full[..., mark_pos] if mark_pos else None
        return x_full[..., keep], x_mark
    # none：移除全部日历列。
    cal_set = {i for i, c in enumerate(schema[:-1]) if is_calendar_col(c)}
    keep = [i for i in range(len(schema)) if i not in cal_set]
    return x_full[..., keep], None


def calendar_ex_mask(ex_mask_full, schema, config):
    """把全 schema 上计算的 ex_mask 调整到 calendar_mode 的 token 顺序。

    ex_mask_full：[..., len(schema)-1]（compute_ex_mask 对完整窗口的输出）。
    日历 token 永不屏蔽（日历列无 __missing 伴随列）；时间标记 token
    不在 ex_mask 内（不进 x_enc）。
    """
    mode = config.get("calendar_mode", "columns")
    if mode == "columns":
        return ex_mask_full
    if mode == "none":
        cal_set = {i for i, c in enumerate(schema[:-1]) if is_calendar_col(c)}
    else:  # time_marks：mark 列移入 x_mark，旗标列仍在 x_enc。
        cal_set = {i for i, c in enumerate(schema[:-1]) if is_mark_col(c)}
    keep = [i for i in range(len(schema) - 1) if i not in cal_set]
    return ex_mask_full[..., keep]


def head_calendar_vector(target_date, head_dim, device):
    """目标日星期 one-hot（预测头日历拼接输入）。

    目标日必为交易日（周一~周五）；one-hot 由日期直接推导，与
    calendar_mode / 日历列开关无关。
    """
    if head_dim != 5:
        raise ValueError(f"head_calendar_dim 应为 5（星期 one-hot），实际 {head_dim}")
    dow = pd.Timestamp(target_date).dayofweek
    if dow > 4:
        raise ValueError(f"目标日 {target_date} 不是交易日（周 {dow + 1}）")
    vec = np.zeros(5, dtype=np.float32)
    vec[dow] = 1.0
    return torch.from_numpy(vec).unsqueeze(0).to(device)


def verify_feature_schema(saved, prep, warn_checkpoint=True):
    """校验 checkpoint 与推理预处理的特征契约。

    新产物（带 feature_schema）严格比对 hash，不匹配即报错；
    旧产物（v8 及以前）缺少该字段时，按列重算并显式告警，避免把
    "缺字段"当成"匹配"而静默兼容。

    warn_checkpoint=False 用于 score 的逐股循环：checkpoint 级告警只在
    第一只股票打印一次，预处理级不一致仍然逐股报告。
    """
    expected = feature_schema_doc(saved["schema"])
    stored = saved.get("feature_schema")
    if stored is None:
        if warn_checkpoint:
            print(
                "[WARN] checkpoint 缺少 feature_schema（旧产物），"
                f"按当前列重算 hash={expected['hash']}；建议重训以固化契约"
            )
    else:
        if stored.get("version") != expected["version"]:
            # 版本升级（如 v1 -> v2 新增日历列）：按存储版本重算 hash，
            # 匹配则显式告警后继续，避免版本常量变化使旧产物失效。
            rebuilt = feature_schema_hash(
                expected["columns"], version=stored.get("version")
            )
            if rebuilt != stored.get("hash"):
                raise ValueError(
                    "特征 schema 版本不一致且按存储版本重算仍不匹配："
                    f"checkpoint={stored.get('version')!r}，"
                    f"当前={expected['version']!r}"
                )
            print(
                "[WARN] 特征 schema 版本升级："
                f"checkpoint={stored.get('version')!r}，"
                f"当前={expected['version']!r}；"
                "按存储版本重算 hash 匹配，继续兼容旧产物"
            )
        elif stored.get("hash") != expected["hash"]:
            raise ValueError(
                "checkpoint 特征 schema hash 与自身列不一致："
                f"{stored.get('hash')} != {expected['hash']}"
            )

    prep_expected = feature_schema_doc(prep["feature_columns"])
    stored_prep = prep.get("feature_schema")
    if stored_prep is None:
        print(
            f"[WARN] 股票 {prep.get('symbol')} 的预处理缺少 feature_schema"
            f"（旧产物），按列重算 hash={prep_expected['hash']}"
        )
    elif stored_prep.get("version") != prep_expected["version"]:
        # 同上：预处理产物版本早于当前代码版本时，按存储版本重算。
        rebuilt = feature_schema_hash(
            prep_expected["columns"], version=stored_prep.get("version")
        )
        if rebuilt != stored_prep.get("hash"):
            raise ValueError(
                f"股票 {prep.get('symbol')} 预处理特征 schema hash 不一致"
                f"（按存储版本 {stored_prep.get('version')!r} 重算）："
                f"{stored_prep.get('hash')} != {rebuilt}"
            )
    elif stored_prep.get("hash") != prep_expected["hash"]:
        raise ValueError(
            f"股票 {prep.get('symbol')} 预处理特征 schema hash 不一致："
            f"{stored_prep.get('hash')} != {prep_expected['hash']}"
        )

    # 预处理列多于 checkpoint schema 属于历史导出差异（如已从 schema
    # 剔除的 px_* 原值列）：align_features 只按 schema 取列，多余列不参与
    # 计算，故显式告警而不拒绝，避免旧 checkpoint 直接失效。
    extra = set(prep["feature_columns"]) - set(saved["schema"])
    if extra:
        print(
            f"[WARN] 股票 {prep.get('symbol')} 预处理包含 schema 之外的列"
            f"{sorted(extra)}（按 schema 对齐后忽略）"
        )
    return expected



@torch.no_grad()
def predict(args, device):
    if not args.checkpoint or not args.data_dir:
        raise ValueError("predict 需要 --checkpoint 和 --data-dir")

    saved = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    adapter = adapter_for_checkpoint(saved.get("format"))

    directory = Path(args.data_dir)
    prep = read_json(directory / "preprocessing.json")
    symbol = prep["symbol"]

    if symbol not in saved["preprocessing"]:
        raise ValueError(
            "该股票不在 checkpoint 的预处理登记中；"
            "不能直接套用其他股票的标准化参数"
        )
    if prep != saved["preprocessing"][symbol]:
        raise ValueError("推理预处理参数与训练期不一致")
    schema_info = verify_feature_schema(saved, prep)

    with np.load(directory / "latest.npz") as latest:
        x_full = align_features(
            latest["X"], prep["feature_columns"], saved["schema"]
        )
        input_date = str(latest["input_date"][0])
        target_date = str(latest["target_date"][0])

    config = saved["model_config"]
    if x_full.shape[:2] != (1, config["seq_len"]):
        raise ValueError(f"推理窗口维度不正确：{x_full.shape}")
    if not np.isfinite(x_full).all():
        raise ValueError("推理窗口包含 NaN/Inf")

    # 日历通道协议跟随 checkpoint（训练时登记，不接受命令行覆盖）；
    # 旧 checkpoint 缺键按历史协议处理并显式提示。
    if "calendar_mode" not in config:
        print(
            "[NOTICE] checkpoint 缺少 calendar_mode（旧产物），按 columns 处理"
        )
    if "head_calendar_dim" not in config:
        print(
            "[NOTICE] checkpoint 缺少 head_calendar_dim（旧产物），按 0 处理"
        )
    x, x_mark = calendar_split(x_full, saved["schema"], config)
    if x.shape != (1, config["seq_len"], config["enc_in"]):
        raise ValueError(f"推理窗口维度不正确：{x.shape}")

    # 掩码协议跟随 checkpoint：训练时启用 ex_mask 的模型，推理时
    # 必须同样屏蔽窗口内整段无观测的外生值 token（在完整窗口上计算，
    # 再按 calendar_mode 调整 token 顺序；日历 token 永不屏蔽）。
    ex_mask = None
    if config.get("ex_mask", False):
        ex_mask_full = compute_ex_mask(
            x_full, exogenous_mask_layout(saved["schema"])
        )
        ex_mask = calendar_ex_mask(ex_mask_full, saved["schema"], config)
        ex_mask = torch.from_numpy(ex_mask).to(device)

    model = adapter.build_model(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    x_mark_t = None
    if x_mark is not None:
        x_mark_t = torch.from_numpy(x_mark).to(device)
    head_calendar = None
    if int(config.get("head_calendar_dim", 0) or 0):
        head_calendar = head_calendar_vector(
            target_date, int(config["head_calendar_dim"]), device
        )
    z = adapter.forward(
        model, torch.from_numpy(x).to(device), ex_mask=ex_mask,
        x_mark=x_mark_t, head_calendar=head_calendar,
    ).item()
    stats = prep["statistics"][TARGET]
    value = z * stats["scale"] + stats["mean"]
    if not np.isfinite(value):
        raise FloatingPointError("预测结果不是有限数值")

    model_id = saved.get("model_id") or sha256_file(args.checkpoint)[:16]
    print(json.dumps({
        "symbol": symbol,
        "as_of_date": input_date,
        "target_date": target_date,
        "model_id": model_id,
        "feature_schema_hash": schema_info["hash"],
        "y_pred": value,
        "predicted_return_percent": value * 100.0,
        "status": PredictionStatus.OK.value,
    }, ensure_ascii=False, indent=2))


@torch.no_grad()
def score(args, saved, device, infer_root):
    """加载全部 latest.npz，批量前向，返回打分明细表。

    （自 predict_daily.py 原样迁出；模型经适配器构造。）
    infer_root：已解析为绝对路径的推理窗口导出目录。
    """
    schema = saved["schema"]
    config = saved["model_config"]

    adapter = adapter_for_checkpoint(saved.get("format"))

    # 掩码协议跟随 checkpoint：训练时启用 ex_mask 的模型，打分时必须
    # 同样屏蔽窗口内整段无观测的外生值 token（判定只看 __missing 标记）。
    use_ex_mask = bool(config.get("ex_mask", False))
    layout = exogenous_mask_layout(schema)

    # 日历通道协议跟随 checkpoint（旧产物缺键按 columns/0 处理；
    # checkpoint 级提示只在第一只股票打印一次）。
    if "calendar_mode" not in config:
        print(
            "[NOTICE] checkpoint 缺少 calendar_mode（旧产物），按 columns 处理"
        )
    if "head_calendar_dim" not in config:
        print(
            "[NOTICE] checkpoint 缺少 head_calendar_dim（旧产物），按 0 处理"
        )

    head_dim = int(config.get("head_calendar_dim", 0) or 0)
    if head_dim not in (0, 5):
        raise ValueError(
            f"head_calendar_dim 应为 0 或 5（星期 one-hot），实际 {head_dim}"
        )

    rows, arrays, marks, masks, heads, symbols = [], [], [], [], [], []
    for index, (symbol, prep) in enumerate(saved["preprocessing"].items()):
        verify_feature_schema(saved, prep, warn_checkpoint=(index == 0))
        path = infer_root / symbol / "latest.npz"
        if not path.exists():
            rows.append({
                "symbol": symbol, "status": PredictionStatus.NO_INFER_OUTPUT.value,
            })
            continue
        with np.load(path, allow_pickle=True) as latest:
            x_full = align_features(
                latest["X"], prep["feature_columns"], schema
            )
            input_date = str(latest["input_date"][0])
            target_date = str(latest["target_date"][0])
        if x_full.shape[:2] != (1, config["seq_len"]):
            rows.append({
                "symbol": symbol, "status": PredictionStatus.BAD_SHAPE.value,
                "input_date": input_date, "target_date": target_date,
                "detail": str(x_full.shape),
            })
            continue
        if not np.isfinite(x_full).all():
            rows.append({
                "symbol": symbol, "status": PredictionStatus.NON_FINITE.value,
                "input_date": input_date, "target_date": target_date,
            })
            continue
        x, x_mark = calendar_split(x_full, schema, config)
        if x.shape != (1, config["seq_len"], config["enc_in"]):
            rows.append({
                "symbol": symbol, "status": PredictionStatus.BAD_SHAPE.value,
                "input_date": input_date, "target_date": target_date,
                "detail": str(x.shape),
            })
            continue
        if head_dim:
            # 目标日星期 one-hot 由目标日期直接推导（与日历列开关无关）。
            dow = pd.Timestamp(target_date).dayofweek
            if dow > 4:
                rows.append({
                    "symbol": symbol,
                    "status": PredictionStatus.BAD_SHAPE.value,
                    "input_date": input_date, "target_date": target_date,
                    "detail": f"目标日不是交易日（周 {dow + 1}）",
                })
                continue
            vec = np.zeros(5, dtype=np.float32)
            vec[dow] = 1.0
        arrays.append(x[0])
        marks.append(x_mark[0] if x_mark is not None else None)
        if use_ex_mask:
            masks.append(
                calendar_ex_mask(compute_ex_mask(x_full, layout), schema, config)[0]
            )
        if head_dim:
            heads.append(vec)
        symbols.append(symbol)
        rows.append({
            "symbol": symbol, "status": PredictionStatus.OK.value,
            "input_date": input_date, "target_date": target_date,
        })
    if not arrays:
        raise RuntimeError("没有可打分的推理窗口")

    model = adapter.build_model(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    has_mark = marks[0] is not None
    preds = []
    for i in range(0, len(arrays), args.batch_size):
        batch = torch.from_numpy(
            np.stack(arrays[i:i + args.batch_size])
        ).to(device)
        ex_mask = None
        if use_ex_mask:
            ex_mask = torch.from_numpy(
                np.stack(masks[i:i + args.batch_size])
            ).to(device)
        x_mark = None
        if has_mark:
            x_mark = torch.from_numpy(
                np.stack(marks[i:i + args.batch_size])
            ).to(device)
        head_calendar = None
        if head_dim:
            head_calendar = torch.from_numpy(
                np.stack(heads[i:i + args.batch_size])
            ).to(device)
        out = adapter.forward(
            model, batch, ex_mask=ex_mask, x_mark=x_mark,
            head_calendar=head_calendar,
        )
        preds.append(out.reshape(-1).cpu().numpy())

    frame = pd.DataFrame(rows).set_index("symbol")
    frame["y_pred"] = np.nan
    for symbol, z in zip(symbols, np.concatenate(preds)):
        stats = saved["preprocessing"][symbol]["statistics"][TARGET]
        value = z * stats["scale"] + stats["mean"]
        frame.loc[symbol, "y_pred"] = value
        frame.loc[symbol, "pred_return_pct"] = value * 100.0
    return frame.reset_index()
