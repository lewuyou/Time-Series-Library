# -*- coding: utf-8 -*-
"""合成数据集验证实验驱动（配套 make_synthetic_dataset.py 产出的数据）。

子命令：
  exp1-oracle   完整公式 Oracle 校验：用模型实际输入窗口（标准化矩阵反推
                原始特征）按生成公式计算无噪声预测，走与 TimeXer 相同的
                测试样本/指标流程。Oracle 应逼近噪声地板，否则先查数据链路。
  exp2-overfit  小样本过拟合测试：固定 256 个窗口、无噪声数据、无 dropout
                /权重衰减/早停，训练足够多轮，看训练误差能否压到接近零。
  exp3-mlp      小型 MLP 对照：与 TimeXer 完全相同的窗口输入与数据划分。

用法示例：
  python synthetic_experiments.py exp1-oracle \
      --data-root dataset/synthetic_001 \
      --pred-csv checkpoints/synthetic_timexer_001/test_predictions.csv
  python synthetic_experiments.py exp2-overfit --data-root dataset/synthetic_nonoise
  python synthetic_experiments.py exp3-mlp --data-root dataset/synthetic_001 \
      --out-dir checkpoints/synthetic_mlp_001
"""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import make_synthetic_dataset as gen
import train_global_timexer as tgt


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    return 1.0 - ((np.asarray(y_pred) - y_true) ** 2).mean() / y_true.var()


# ---------------------------------------------------------------- 实验一


def cmd_exp1(args):
    """完整公式 Oracle：与 TimeXer 同样本、同指标流程直接对照。"""
    pred = pd.read_csv(args.pred_csv)
    root = Path(args.data_root)
    spec = json.loads((root / "export_config.json").read_text())
    dgp = spec["dgp"]
    noise_std = dgp["noise_std"]

    oracle, truths = [], []
    checks = {"end+1==target": True, "input<target": True}
    for symbol, part in pred.groupby("symbol"):
        d = root / symbol
        prep = json.loads((d / "preprocessing.json").read_text())
        cols, st = prep["feature_columns"], prep["statistics"]
        matrix = np.load(d / "feature_matrix.npy")

        def raw(column):
            s = st[column]
            return (
                matrix[:, cols.index(column)].astype(np.float64)
                * s["scale"]
                + s["mean"]
            )

        signal = gen.dgp_signal(
            {
                "syn_momentum": raw("syn_momentum"),
                "syn_volatility": raw("syn_volatility"),
                "baidu_search_all": raw("baidu_search_all"),
                "rzrq_fin_tval_ratio": raw("rzrq_fin_tval_ratio"),
                "macro_close_SYNSRC_A": raw("macro_close_SYNSRC_A"),
            },
            gen.DGP_COEFFS,
            task=dgp.get("task", "all"),
        )
        dates = np.load(d / "dates.npy").astype("datetime64[D]")
        position = {str(day): i for i, day in enumerate(dates)}
        samples = pd.read_csv(d / "samples.csv")
        by_target = {
            (int(r.end_pos), int(r.target_pos)): r
            for _, r in samples.iterrows()
        }
        for _, row in part.iterrows():
            # 对齐检查：样本表里 end_pos+1 == target_pos；
            # oracle 只取 signal[t=end_pos]（滚动均值仅用 t-9..t 的特征）。
            t = position[row.target_date] - 1
            rec = by_target.get((t, t + 1))
            if rec is None:
                raise AssertionError(f"{symbol} 样本对齐失败 @ {row.target_date}")
            oracle.append(float(signal[t]))
            truths.append(float(row.y_true))
            checks["end+1==target"] &= int(rec.end_pos) + 1 == int(rec.target_pos)
            checks["input<target"] &= str(rec.input_date) < str(rec.target_date)

    oracle = np.array(oracle)
    y = np.array(truths)
    model_pred = pred.sort_values(["symbol", "target_date"]).y_pred.to_numpy()
    residual = y - oracle

    print("== 实验1：完整公式 Oracle 校验 ==")
    print(f"任务: {dgp.get('task', 'all')}  样本数: {len(y)}")
    print(f"对齐检查: {checks}")
    print(f"noise_std (设计): {noise_std:.4f}")
    print(f"残差 std (实际):  {residual.std():.4f}  "
          f"(Oracle 残差应≈标签噪声)")
    print(f"残差均值: {residual.mean():+.5f} (应≈0，检验标签偏移)")
    print()
    rows = [
        ("Oracle 完整公式", oracle),
        ("TimeXer (对照)", model_pred),
    ]
    print(f'{"预测源":<18}{"MSE":>10}{"R2":>9}{"Pearson":>9}')
    for name, p in rows:
        mse = ((p - y) ** 2).mean()
        print(
            f"{name:<18}{mse:>10.4f}{r2_score(y, p):>9.3f}"
            f"{np.corrcoef(p, y)[0, 1]:>9.3f}"
        )
    floor = noise_std ** 2
    achieved = 1 - ((oracle - y) ** 2).mean() / y.var()
    print()
    print(f"有限样本噪声地板 MSE={floor:.4f} → Oracle 有限样本 R2={achieved:.3f}")
    print(f"理论总体 R2 上限={dgp['theoretical_max_r2']:.3f} (参考值)")


# ---------------------------------------------------------------- 实验二


def cmd_exp2(args):
    """小样本过拟合：能否把固定 256 个无噪声窗口的训练误差压到≈0。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records, schema = tgt.load_catalog(args.data_root)
    train_ds = tgt.PanelWindows(records, schema, "train")

    # 固定子集：每股前 N//n 只样本（确定性，无随机性）。
    per_stock = args.samples // len(records)
    chosen = []
    counts = [0] * len(records)
    for i in range(len(train_ds)):
        sid = bisect.bisect_right(train_ds.ends, i)
        if counts[sid] < per_stock:
            chosen.append(i)
            counts[sid] += 1
        if all(c >= per_stock for c in counts):
            break

    xs, ys = [], []
    for i in chosen:
        x, y, *_ = train_ds[i]
        xs.append(x)
        ys.append(y)
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))
    print(f"过拟合子集: {len(x)} 窗口 (每股 {per_stock}), "
          f"输入 {tuple(x.shape)}, 设备 {device}")

    config = {
        "task_name": "long_term_forecast", "features": "MS",
        "seq_len": train_ds.seq_len, "pred_len": 1, "label_len": 0,
        "enc_in": len(schema), "c_out": 1,
        "d_model": args.d_model, "d_ff": args.d_model * 2,
        "e_layers": 2, "n_heads": 4, "factor": 1,
        "dropout": 0.0,            # 关闭全部正则
        "embed": "fixed", "freq": "d", "activation": "gelu",
        "use_norm": 0, "patch_len": args.patch_len, "ex_mask": False,
    }
    model = tgt.build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0
    )
    loader = DataLoader(
        TensorDataset(x, y), batch_size=args.batch_size, shuffle=True
    )

    print("== 实验2：小样本过拟合测试（无噪声标签, 无正则/早停）==")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = (model(xb, None, None, None) - yb).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
        if epoch % 50 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  train MSE(scaled) = {total / count:.6e}")

    final = total / count
    scales = [
        r["prep"]["statistics"]["stock_return"]["scale"] for r in records
    ]
    rmse_raw = np.sqrt(final) * float(np.mean(scales))
    print()
    print(f"最终训练 MSE(scaled)={final:.3e}  "
          f"≈无噪声地板应为 ~0；折算 RMSE(raw)≈{rmse_raw:.5f}")
    print("若无法压低 → 优先排查实现/优化/信息通路，而非归因架构容量。")


# ---------------------------------------------------------------- 实验三


def cmd_exp3(args):
    """小型 MLP 对照：与 TimeXer 相同的窗口、划分、早停协议。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, schema = tgt.load_catalog(args.data_root)
    train_ds = tgt.PanelWindows(records, schema, "train")
    val_ds = tgt.PanelWindows(records, schema, "val", allow_missing=True)
    test_ds = tgt.PanelWindows(records, schema, "test", allow_missing=True)

    def tensors(ds):
        xs, ys = [], []
        for i in range(len(ds)):
            x, y, *_ = ds[i]
            xs.append(x)
            ys.append(y)
        return (torch.from_numpy(np.stack(xs)),
                torch.from_numpy(np.stack(ys)).reshape(-1))

    xtr, ytr = tensors(train_ds)
    xv, yv = tensors(val_ds)
    xte, yte = tensors(test_ds)

    c = xtr.shape[1] * xtr.shape[2]
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(c, 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True
    )

    print(f"== 实验3：MLP 对照 == 输入展平维度={c}, 设备={device}")
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = (model(xb).squeeze(-1) - yb).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = (
                (model(xv.to(device)).squeeze(-1) - yv.to(device))
                .square().mean().item()
            )
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= args.patience:
                print(f"早停 @ epoch {epoch}, best val(scaled)={best:.5f}")
                break
    model.load_state_dict(best_state)
    model.eval()

    # 测试集：与 test_predictions 相同的反标准化与 pooled 指标。
    means, scales = [], []
    for i in range(len(test_ds)):
        sid = bisect.bisect_right(test_ds.ends, i)
        r = records[sid]
        s = r["prep"]["statistics"]["stock_return"]
        means.append(s["mean"])
        scales.append(s["scale"])
    means, scales = np.array(means), np.array(scales)
    y_true = yte.numpy().astype(np.float64)
    with torch.no_grad():
        p_scaled = (
            model(xte.to(device)).squeeze(-1).cpu().numpy()
        )
    y_hat = p_scaled * scales + means
    y_raw = y_true * scales + means

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "mlp_flatten",
        "n_test": int(len(y_raw)),
        "mse": float(((y_hat - y_raw) ** 2).mean()),
        "r2": float(r2_score(y_raw, y_hat)),
        "pearson": float(np.corrcoef(y_hat, y_raw)[0, 1]),
        "mae": float(np.abs(y_hat - y_raw).mean()),
        "best_val_scaled_mse": best,
    }
    (out / "mlp_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("exp1-oracle")
    p1.add_argument("--data-root", required=True)
    p1.add_argument("--pred-csv", required=True)
    p1.set_defaults(func=cmd_exp1)

    p2 = sub.add_parser("exp2-overfit")
    p2.add_argument("--data-root", required=True)
    p2.add_argument("--samples", type=int, default=256)
    p2.add_argument("--epochs", type=int, default=500)
    p2.add_argument("--batch-size", type=int, default=64)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--d-model", type=int, default=64)
    p2.add_argument("--patch-len", type=int, default=10)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--device", default="auto")
    p2.set_defaults(func=cmd_exp2)

    p3 = sub.add_parser("exp3-mlp")
    p3.add_argument("--data-root", required=True)
    p3.add_argument("--out-dir", required=True)
    p3.add_argument("--epochs", type=int, default=80)
    p3.add_argument("--patience", type=int, default=8)
    p3.add_argument("--batch-size", type=int, default=256)
    p3.add_argument("--seed", type=int, default=42)
    p3.set_defaults(func=cmd_exp3)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
