"""统一训练器：模型无关的 epoch 循环与测试预测。

自 train_global_timexer.py 原样迁出（run_epoch / test_predictions），
模型前向改为经适配器 adapter.forward 转发（TimeXer 行为与原脚本
逐位一致：model(x, None, None, None, ex_mask=ex_mask)）。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from stock.data.schema import TARGET

LOSS_TYPES = ("mse", "huber")


def per_sample_loss(diff, loss_type="mse", huber_delta=1.0):
    """逐样本损失：输入 (pred - y)，形状 (B, ...)，返回 (B,)。

    mse：逐元素平方后按样本维平均（历史默认口径）。
    huber：逐元素 Huber（delta 可配）后按样本维平均；目标为按股
    标准化后的收益率，尺度约 1，delta 默认 1.0。
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"不支持的损失类型：{loss_type}")
    if loss_type == "huber":
        element = F.huber_loss(
            diff, torch.zeros_like(diff), reduction="none",
            delta=float(huber_delta),
        )
    else:
        element = diff.square()
    return element.reshape(len(diff), -1).mean(1)


def run_epoch(adapter, model, loader, device, optimizer=None,
              use_ex_mask=False, loss_type="mse", huber_delta=1.0):
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
        for x, y, sid, _, _, ex_mask, x_mark, head_cal in loader:
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

            # x_mark 时间标记通道：0 列（columns/none 模式）时传 None。
            if x_mark.shape[-1]:
                x_mark = x_mark.to(device, non_blocking=True)
            else:
                x_mark = None

            # head_cal 数据集恒产出；是否使用由模型配置
            # （head_calendar_dim）决定，适配器内部判定。
            head_cal = head_cal.to(device, non_blocking=True)

            pred = adapter.forward(
                model, x, ex_mask=ex_mask, x_mark=x_mark,
                head_calendar=head_cal,
            )
            if pred.shape != y.shape:
                raise ValueError(f"输出 {pred.shape} != 标签 {y.shape}")

            per_sample = per_sample_loss(
                pred - y, loss_type=loss_type, huber_delta=huber_delta
            )
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

    for x, y, sid, target, y_raw, ex_mask, x_mark, head_cal in loader:
        # 与训练协议一致：use_ex_mask=False 时掩码不进入模型。
        if use_ex_mask:
            ex_mask = ex_mask.to(device, non_blocking=True)
        else:
            ex_mask = None
        if x_mark.shape[-1]:
            x_mark = x_mark.to(device, non_blocking=True)
        else:
            x_mark = None
        output = adapter.forward(
            model, x.to(device), ex_mask=ex_mask, x_mark=x_mark,
            head_calendar=head_cal.to(device, non_blocking=True),
        )
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
