# -*- coding: utf-8 -*-
"""TimeXer 单股票下一交易日收益率训练（独立训练循环）。

为什么不用 run.py / exp 循环：
    StockTimeXerDataset 返回 (x, y) 二元组，而库内训练/测试循环要求
    (batch_x, batch_y, batch_x_mark, batch_y_mark) 四元组，测试逻辑还依赖
    Dataset_Custom 风格的 scale / inverse_transform 接口。本任务的样本
    切分与标准化由 dataset.py 导出文件决定，因此只复用 models.TimeXer
    模型本体，训练循环独立实现。

关键设计（对应外部审查结论）：
    1. features="MS"：TimeXer 把输入最后一列（stock_return）作为内生
       目标序列，其余列作为外生变量；c_out=1 不能替代 MS 的分支选择。
    2. patch_len 必须整除 seq_len：内生分支用不重叠 unfold 切 patch，
       60/16 只覆盖 0~47、丢弃最近 12 天目标历史。默认自动取 4~16 内
       seq_len 的最大因子，可用 --patch-len 覆盖。
    3. label_len=0：本任务标签只有未来 1 天，没有可给 decoder 的历史
       上下文；TimeXer 本身不使用 decoder 输入，但固定为 0 防止将来换
       模型时把未来标签拼进 dec_inp。
    4. 每轮只在验证集上早停；训练期间不计算、不打印测试损失，模型
       选定后测试集只评估一次，避免用测试表现调参。
    5. 评估在简单收益率空间（对数标签自动 expm1）进行，并同时报告
       零预测、训练期均值、持续性（昨日收益率）三个基准。
    6. 默认小模型（d_model=64 / d_ff=128 / e_layers=1）：单股日频样本
       通常只有数千条，库默认 d_model=512 / d_ff=2048 极易过拟合。
    7. 预测的是统计口径收益率，不等于可实现交易收益（t+1 早 08:30
       出信号后无法按 t 收盘价成交）。

用法：
    python train_timexer.py --data-dir dataset/timexer_run_001/601398 \
        --out-dir checkpoints/stock_timexer_001

输出（--out-dir）：
    checkpoint.pth          早停保存的最优模型
    metrics.json            训练过程与测试评估结果（含基准对比）
    test_predictions.csv    测试集逐日预测与真值（简单收益率空间）
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import StockTimeXerDataset
from models.TimeXer import Model as TimeXer
from utils.tools import EarlyStopping


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="dataset.py 导出的股票目录")
    p.add_argument("--out-dir", default=None, help="输出目录，默认 checkpoints/<股票>_sl<seq_len>")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--e-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use-norm", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--patch-len", type=int, default=None,
        help="内生 patch 长度，必须整除 seq_len；缺省自动取 4~16 内 "
             "seq_len 的最大因子",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def default_patch_len(seq_len):
    """取 4~16 范围内 seq_len 的最大因子；没有则退到最大因子。"""
    divisors = [
        d for d in range(1, min(seq_len, 16) + 1) if seq_len % d == 0
    ]
    preferred = [d for d in divisors if d >= 4]
    return max(preferred) if preferred else max(divisors)


def build_model(args, seq_len, enc_in, patch_len):
    configs = argparse.Namespace(
        task_name="long_term_forecast",
        features="MS",       # 末列为内生目标，其余为外生变量
        seq_len=seq_len,
        pred_len=1,
        label_len=0,
        enc_in=enc_in,
        c_out=1,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        factor=1,
        dropout=args.dropout,
        embed="fixed",
        freq="h",
        activation="gelu",
        use_norm=args.use_norm,
        patch_len=patch_len,
    )
    return TimeXer(configs).float()


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x, None, None, None)
            if outputs.shape != batch_y.shape:
                raise AssertionError(
                    f"模型输出形状 {tuple(outputs.shape)} 与标签 "
                    f"{tuple(batch_y.shape)} 不一致"
                )
            loss = criterion(outputs, batch_y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch_y.shape[0]
            count += batch_y.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """返回 (pred_scaled, true_scaled, last_scaled) 的展平数组。

    last_scaled 为输入窗口最后一行的 stock_return（标准化），即持续性
    基准所需的"昨日收益率"。
    """
    model.eval()
    preds, trues, lasts = [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.float().to(device)
        outputs = model(batch_x, None, None, None)
        if outputs.shape != batch_y.shape:
            raise AssertionError(
                f"模型输出形状 {tuple(outputs.shape)} 与标签 "
                f"{tuple(batch_y.shape)} 不一致"
            )
        preds.append(outputs.squeeze().cpu().numpy())
        trues.append(batch_y.squeeze().numpy())
        lasts.append(batch_x[:, -1, -1].cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(trues),
        np.concatenate(lasts),
    )


def return_metrics(y_true, y_pred):
    err = y_pred - y_true
    nonzero = y_true != 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mse": float(np.mean(err ** 2)),
        "directional_accuracy": (
            float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])))
            if nonzero.any() else None
        ),
        "n": int(len(y_true)),
    }


def to_simple(ds, values):
    return np.asarray(ds.to_simple_return(np.asarray(values, dtype=float)))


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs / batch-size / patience 必须为正数")
    if args.d_model % args.n_heads != 0:
        raise ValueError(
            f"d_model={args.d_model} 必须能被 n_heads={args.n_heads} 整除"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )

    ds_train = StockTimeXerDataset(args.data_dir, split="train")
    ds_val = StockTimeXerDataset(args.data_dir, split="val")
    ds_test = StockTimeXerDataset(args.data_dir, split="test")

    seq_len, enc_in = ds_train.seq_len, ds_train.feature_count
    patch_len = args.patch_len or default_patch_len(seq_len)
    if seq_len % patch_len != 0:
        divisors = [
            d for d in range(1, seq_len + 1) if seq_len % d == 0
        ]
        raise ValueError(
            f"patch_len={patch_len} 不整除 seq_len={seq_len}：TimeXer 内生"
            "分支会丢弃窗口尾部历史。可用因子："
            f"{[d for d in divisors if d <= 32]}"
        )

    out_dir = Path(
        args.out_dir
        or f"checkpoints/{ds_train.preprocessing['symbol']}_sl{seq_len}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"data={args.data_dir} | symbol={ds_train.preprocessing['symbol']} | "
        f"seq_len={seq_len} enc_in={enc_in} patch_len={patch_len} "
        f"patch_num={seq_len // patch_len} | "
        f"train/val/test={len(ds_train)}/{len(ds_val)}/{len(ds_test)} | "
        f"return_type={ds_train.return_type} | device={device}"
    )

    model = build_model(args, seq_len, enc_in, patch_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params={n_params:,} | d_model={args.d_model} "
          f"d_ff={args.d_ff} e_layers={args.e_layers} use_norm={args.use_norm}")

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            criterion=criterion,
        )
        # 只用验证集早停；训练期间不触碰测试集。
        val_loss = run_epoch(
            model, val_loader, device, criterion=criterion,
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": round(time.time() - t0, 2),
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | {time.time() - t0:.1f}s"
        )
        early_stopping(val_loss, model, str(out_dir))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(
        torch.load(out_dir / "checkpoint.pth", map_location=device)
    )

    # ---- 测试集一次性评估（简单收益率空间）+ 基准对比 ----
    pred_s, true_s, last_s = collect_predictions(model, test_loader, device)
    y_pred = to_simple(ds_test, pred_s)
    y_true = to_simple(ds_test, true_s)

    zero_pred = np.zeros_like(y_true)
    train_mean_label = float(ds_train.samples["y_raw"].mean())
    mean_pred = np.full_like(y_true, train_mean_label)
    if ds_train.return_type == "log":
        mean_pred = np.expm1(mean_pred)
    persist_pred = to_simple(ds_test, last_s)

    results = {
        "data_dir": str(args.data_dir),
        "symbol": ds_train.preprocessing["symbol"],
        "seq_len": seq_len,
        "patch_len": patch_len,
        "enc_in": enc_in,
        "return_type": ds_train.return_type,
        "params": n_params,
        "model_config": {
            "d_model": args.d_model, "d_ff": args.d_ff,
            "e_layers": args.e_layers, "n_heads": args.n_heads,
            "dropout": args.dropout, "use_norm": args.use_norm,
        },
        "train_config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "patience": args.patience, "seed": args.seed,
            "best_epoch": int(min(history, key=lambda h: h["val_loss"])["epoch"])
            if history else None,
        },
        "history": history,
        "test_metrics_scaled_mse": {
            "model": float(np.mean((pred_s - true_s) ** 2)),
        },
        "test_metrics_simple_return": {
            "model": return_metrics(y_true, y_pred),
            "baseline_zero": return_metrics(y_true, zero_pred),
            "baseline_train_mean": return_metrics(y_true, mean_pred),
            "baseline_persistence": return_metrics(y_true, persist_pred),
        },
        "note": (
            "统计口径收益率预测；t+1 早 08:30 出信号后无法按 t 收盘价成交，"
            "不能直接换算为可实现交易收益"
        ),
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    json_write(out_dir / "metrics.json", results)

    pd.DataFrame({
        "target_date": ds_test.samples["target_date"].astype(str),
        "y_true_simple": y_true,
        "y_pred_simple": y_pred,
        "baseline_zero": zero_pred,
        "baseline_train_mean": mean_pred,
        "baseline_persistence": persist_pred,
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    print("\n测试集评估（简单收益率空间，模型在验证集上选定）：")
    print(f"{'name':<20}{'MAE':>10}{'RMSE':>10}{'DirAcc':>10}")
    for name, pred in [
        ("model", y_pred),
        ("baseline_zero", zero_pred),
        ("baseline_train_mean", mean_pred),
        ("baseline_persistence", persist_pred),
    ]:
        m = results["test_metrics_simple_return"][name]
        dir_acc = (
            f"{m['directional_accuracy']:.4f}"
            if m["directional_accuracy"] is not None else "n/a"
        )
        print(f"{name:<20}{m['mae']:>10.6f}{m['rmse']:>10.6f}{dir_acc:>10}")

    print(f"\n[OK] 训练完成：{time.time() - start:.1f}s，结果保存于 {out_dir}")
    return 0


def json_write(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
