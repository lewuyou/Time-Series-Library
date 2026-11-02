"""统一训练器：模型无关的 epoch 循环与测试预测。

自 train_global_timexer.py 原样迁出（run_epoch / test_predictions），
模型前向改为经适配器 adapter.forward 转发（TimeXer 行为与原脚本
逐位一致：model(x, None, None, None, ex_mask=ex_mask)）。
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from stock.data.schema import TARGET


def run_epoch(adapter, model, loader, device, optimizer=None,
              use_ex_mask=False):
    training = optimizer is not None
    model.train(training)

    weights = torch.as_tensor(
        loader.dataset.stock_weights,
        dtype=torch.float32,
        device=device,
    )
    total = 0.0
    count = 0

    with torch.set_grad_enabled(training):
        for x, y, sid, _, _, ex_mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sid = sid.to(device, non_blocking=True)

            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("输入/标签存在非有限值")

            # ex_mask 数据集恒产出；use_ex_mask=False 时不传入模型，
            # 与既有"填补 + 缺失标记"协议逐位一致。
            if use_ex_mask:
                ex_mask = ex_mask.to(device, non_blocking=True)
            else:
                ex_mask = None

            pred = adapter.forward(model, x, ex_mask=ex_mask)
            if pred.shape != y.shape:
                raise ValueError(f"输出 {pred.shape} != 标签 {y.shape}")

            per_sample = (pred - y).square().reshape(len(x), -1).mean(1)
            loss = (per_sample * weights[sid]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("loss 为 NaN/Inf")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 梯度非有限时立即报错，而不是让 NaN/Inf 静默进入参数。
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()

            total += loss.detach().item() * len(x)
            count += len(x)

    return total / count


@torch.no_grad()
def test_predictions(adapter, model, loader, records, device,
                     use_ex_mask=False):
    model.eval()
    chunks = []

    for x, y, sid, target, y_raw, ex_mask in loader:
        # 与训练协议一致：use_ex_mask=False 时掩码不进入模型。
        if use_ex_mask:
            ex_mask = ex_mask.to(device, non_blocking=True)
        else:
            ex_mask = None
        output = adapter.forward(model, x.to(device), ex_mask=ex_mask)
        if output.shape != y.shape:
            raise ValueError("测试输出维度不正确")

        pred_scaled = output.cpu().numpy().reshape(-1)
        last_scaled = x[:, -1, -1].numpy()
        # y_true 直接用样本表 float64 原始标签（与两阶段协议的
        # y_true 一致性检查对齐），不经 float32 标签反标准化重建。
        y_true = y_raw.numpy().astype(np.float64)

        ids = sid.numpy()
        target = target.numpy()
        means = np.array([
            records[i]["prep"]["statistics"][TARGET]["mean"]
            for i in ids
        ])
        scales = np.array([
            records[i]["prep"]["statistics"][TARGET]["scale"]
            for i in ids
        ])

        chunks.append(pd.DataFrame({
            "symbol": [records[i]["symbol"] for i in ids],
            "target_date": [
                str(records[i]["dates"][j].astype("datetime64[D]"))
                for i, j in zip(ids, target)
            ],
            "y_true": y_true,
            "y_pred": pred_scaled * scales + means,
            "baseline_zero": np.zeros(len(ids)),
            "baseline_mean": [records[i]["train_mean"] for i in ids],
            "baseline_persistence": last_scaled * scales + means,
        }))

    result = pd.concat(chunks, ignore_index=True)
    numeric = result.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise FloatingPointError("测试预测包含 NaN/Inf")
    return result
