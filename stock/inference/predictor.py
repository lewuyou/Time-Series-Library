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
)
from stock.utils import read_json


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

    print(json.dumps({
        "symbol": symbol,
        "input_date": input_date,
        "target_date": target_date,
        "predicted_simple_return": value,
        "predicted_return_percent": value * 100.0,
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
    for symbol, prep in saved["preprocessing"].items():
        path = infer_root / symbol / "latest.npz"
        if not path.exists():
            rows.append({"symbol": symbol, "note": "no_infer_output"})
            continue
        with np.load(path, allow_pickle=True) as latest:
            x = align_features(latest["X"], prep["feature_columns"], schema)
            input_date = str(latest["input_date"][0])
            target_date = str(latest["target_date"][0])
        if x.shape != (1, config["seq_len"], config["enc_in"]):
            rows.append({"symbol": symbol, "note": f"bad_shape_{x.shape}"})
            continue
        if not np.isfinite(x).all():
            rows.append({"symbol": symbol, "note": "non_finite"})
            continue
        arrays.append(x[0])
        if use_ex_mask:
            masks.append(compute_ex_mask(x, layout)[0])
        symbols.append(symbol)
        rows.append({
            "symbol": symbol, "note": "ok",
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
    for symbol, z in zip(symbols, np.concatenate(preds)):
        stats = saved["preprocessing"][symbol]["statistics"][TARGET]
        frame.loc[symbol, "pred_return_pct"] = (
            z * stats["scale"] + stats["mean"]
        ) * 100.0
    return frame.reset_index()
