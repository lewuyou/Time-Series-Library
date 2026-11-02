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
)
from stock.inference.contracts import PredictionStatus
from stock.utils import read_json, sha256_file


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
            raise ValueError(
                "特征 schema 版本不一致："
                f"checkpoint={stored.get('version')!r}，当前={expected['version']!r}"
            )
        if stored.get("hash") != expected["hash"]:
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
        x = align_features(
            latest["X"], prep["feature_columns"], saved["schema"]
        )
        input_date = str(latest["input_date"][0])
        target_date = str(latest["target_date"][0])

    config = saved["model_config"]
    if x.shape != (1, config["seq_len"], config["enc_in"]):
        raise ValueError(f"推理窗口维度不正确：{x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("推理窗口包含 NaN/Inf")

    # 掩码协议跟随 checkpoint：训练时启用 ex_mask 的模型，推理时
    # 必须同样屏蔽窗口内整段无观测的外生值 token。
    ex_mask = None
    if config.get("ex_mask", False):
        ex_mask = compute_ex_mask(x, exogenous_mask_layout(saved["schema"]))
        ex_mask = torch.from_numpy(ex_mask).to(device)

    model = adapter.build_model(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    z = adapter.forward(
        model, torch.from_numpy(x).to(device), ex_mask=ex_mask
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

    rows, arrays, masks, symbols = [], [], [], []
    for index, (symbol, prep) in enumerate(saved["preprocessing"].items()):
        verify_feature_schema(saved, prep, warn_checkpoint=(index == 0))
        path = infer_root / symbol / "latest.npz"
        if not path.exists():
            rows.append({
                "symbol": symbol, "status": PredictionStatus.NO_INFER_OUTPUT.value,
            })
            continue
        with np.load(path, allow_pickle=True) as latest:
            x = align_features(latest["X"], prep["feature_columns"], schema)
            input_date = str(latest["input_date"][0])
            target_date = str(latest["target_date"][0])
        if x.shape != (1, config["seq_len"], config["enc_in"]):
            rows.append({
                "symbol": symbol, "status": PredictionStatus.BAD_SHAPE.value,
                "input_date": input_date, "target_date": target_date,
                "detail": str(x.shape),
            })
            continue
        if not np.isfinite(x).all():
            rows.append({
                "symbol": symbol, "status": PredictionStatus.NON_FINITE.value,
                "input_date": input_date, "target_date": target_date,
            })
            continue
        arrays.append(x[0])
        if use_ex_mask:
            masks.append(compute_ex_mask(x, layout)[0])
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
        out = adapter.forward(model, batch, ex_mask=ex_mask)
        preds.append(out.reshape(-1).cpu().numpy())

    frame = pd.DataFrame(rows).set_index("symbol")
    frame["y_pred"] = np.nan
    for symbol, z in zip(symbols, np.concatenate(preds)):
        stats = saved["preprocessing"][symbol]["statistics"][TARGET]
        value = z * stats["scale"] + stats["mean"]
        frame.loc[symbol, "y_pred"] = value
        frame.loc[symbol, "pred_return_pct"] = value * 100.0
    return frame.reset_index()
