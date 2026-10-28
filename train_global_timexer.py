# -*- coding: utf-8 -*-
"""TimeXer 多股票共享模型：独立窗口，联合训练（完整流程第 3/4 步）。

完整流程（第 1~4 步）：
    第 1 步 粗筛股票池：python select_universe.py --db ../AutoTS/download/autots.duckdb \
        --out dataset/global_001/universe.csv --top-mv 3000 \
        --min-years 3 --baidu-fresh-days 30
    第 2 步 批量导出：python export_global.py --db ../AutoTS/download/autots.duckdb \
        --universe-csv dataset/global_001/universe.csv \
        --data-root dataset/global_001 --workers 4 \
        --seq-len 60 --test-days 30 --val-days 60
        （建议先抽查：python dataset.py --mode validate --data-dir dataset/global_001/601398）
    第 3 步 联合训练（本脚本 --mode train）：
        python train_global_timexer.py --mode train \
            --data-root dataset/global_001 \
            --out-dir checkpoints/global_timexer_001 \
            --epochs 30 --patience 5 --batch-size 256 --workers 0 \
            --d-model 64 --e-layers 2 --patch-len 10 --device auto
    第 4 步 推理预测（本脚本 --mode predict）：
        python dataset.py --mode infer --db ../AutoTS/download/autots.duckdb \
            --preprocessor dataset/global_001/601398/preprocessing.json \
            --out-dir dataset/global_infer_001 --fail-on-history-drift
        python train_global_timexer.py --mode predict \
            --checkpoint checkpoints/global_timexer_001/checkpoint.pth \
            --data-dir dataset/global_infer_001/601398 --device auto

训练输出（--out-dir）：
    checkpoint.pth                早停保存的最优共享模型（含每股预处理登记）
    history.json                  逐轮 train/val 股票等权标准化 MSE
    schema.json                   统一特征 schema
    test_predictions.csv          测试集逐样本预测与真值 + 3 个基准
    test_metrics.json             汇总指标（pooled / 股票等权 / 日度 RankIC）
    test_metrics_by_stock.csv     按股票 × 预测源的指标明细

核心设计：
    1. 共享模型参数一份、统一特征 schema 一份、每只股票训练期预处理参数
       各自一份（已嵌入 checkpoint）。predict 时对同一股票必须复用训练期
       scaler，不能每次预测都用最新全量历史重新拟合。
    2. 全局样本索引只定位到某股票内部，绝不跨股票切片；可以混合不同
       股票的窗口组成 batch，但不能混接股票时间轴。
    3. 每个 epoch 所有股票总权重相同（股票等权），避免训练窗口更多的
       股票主导损失（控制的是样本数量而非市值大小；AdamW 与梯度裁剪
       仍会影响实际参数更新）。
    4. 每轮只在验证集上早停；测试集仅在模型选定后评估一次。
    5. patch-len 必须整除 seq_len（TimeXer 内生分支用不重叠 patch）。
    6. 第一轮建议 --workers 0 便于定位数据问题；大量股票随机访问时
       磁盘、系统缓存与 DataLoader 进程开销可能成为瓶颈，不是 worker
       越多越快。
    7. checkpoint 只保存最优模型（无优化器/随机数状态），不是训练恢复
       文件；"断点续跑"仅指导出阶段（export_config.json），输出目录已有
       checkpoint 时会直接拒绝训练。
    8. predict 仅支持 checkpoint 中登记过的股票（每股训练期 scaler），
       是"当前训练股票池共享一个模型"，不是任意新股票的零样本预测。

已知局限（详见使用说明第九节）：
    1. "按假设发布时间对齐"不是真实 point-in-time：百度指数实际延迟、
       两融实际发布时刻、跨时区行情、宏观修订、历史复权更新等需人工核实。
    2. 股票池按当前市值/当前存续/当前简称与最新百度新鲜度筛选，存在
       幸存者偏差；只能回答"当前这批股票在留出日期上表现如何"，不能
       回答历史任意时点全市场股票池的真实投资收益。
    3. 默认 30 个测试日只是初步检查：同一天股票高度相关，不能当作
       大量独立观测；正式验证需扩大测试跨度、滚动切分、分行业/市值/
       波动率评估；若要声称对训练中没见过的股票有效，需额外做整只
       股票留出测试。
    4. 预测的是统计口径的下一交易日收益率（t+1 早 08:30 出信号后已
       无法按 t 收盘价成交），不等于可实现策略收益；回测需重新明确
       成交时点、持有期限、费用、停牌与涨跌停约束。
    5. 建议固定项目 git commit，避免官方 TimeXer.py / embedding 实现
       变化后与 checkpoint 结构不兼容。
"""

import argparse
import bisect
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.TimeXer import Model as TimeXer


TARGET = "stock_return"
DROP_BASES = {"px_open", "px_close", "px_high", "px_low"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def base_name(column):
    return column.split("__", 1)[0]


def align_features(x, source_columns, schema):
    """重排列；被某股票过滤掉的特征作为缺失，而不是挪用其他列。"""
    result = np.zeros((*x.shape[:-1], len(schema)), dtype=np.float32)
    missing_positions = [
        i for i, c in enumerate(schema) if c.endswith("__missing")
    ]
    result[..., missing_positions] = 1.0

    source = {c: i for i, c in enumerate(source_columns)}
    dst = [i for i, c in enumerate(schema) if c in source]
    src = [source[schema[i]] for i in dst]
    result[..., dst] = x[..., src]
    return result


def load_catalog(data_root):
    root = Path(data_root)
    report = pd.read_csv(
        root / "export_results.csv", dtype={"symbol": str}
    )
    required = {"symbol", "status"}
    if not required.issubset(report.columns):
        raise ValueError("export_results.csv 缺少必要字段（symbol/status）")
    if report["symbol"].duplicated().any():
        raise ValueError("export_results.csv 存在重复股票")
    if (report["status"] == "error").any():
        raise ValueError(
            "导出报告仍有 error，请先修复导出，避免静默训练部分股票池"
        )

    # 与导出任务登记对比，确保时间边界与窗口长度和本次 data-root
    # 一致，而不只是股票之间相互一致。
    spec = read_json(root / "export_config.json")
    for key in ("train_end", "val_end", "seq_len"):
        if key not in spec:
            raise ValueError(f"export_config.json 缺少字段 {key}")

    symbols = sorted(
        report.loc[report.status == "ok", "symbol"].unique().tolist()
    )
    if not symbols:
        raise ValueError("没有导出复检通过的股票")

    records = []
    reference = None

    for symbol in symbols:
        directory = root / symbol
        prep = read_json(directory / "preprocessing.json")
        columns = prep["feature_columns"]

        if prep["symbol"] != symbol:
            raise ValueError(f"{symbol}: 预处理股票代码不匹配")
        if columns[-1] != TARGET or len(columns) != len(set(columns)):
            raise ValueError(f"{symbol}: 特征列定义异常")
        if prep["config"]["return_type"] != "simple":
            raise ValueError("本脚本要求 simple 收益率，请重新导出")
        if prep["train_end"] != spec["train_end"]:
            raise ValueError(f"{symbol}: train_end 与导出任务登记不一致")
        if prep["val_end"] != spec["val_end"]:
            raise ValueError(f"{symbol}: val_end 与导出任务登记不一致")
        if prep["config"]["seq_len"] != spec["seq_len"]:
            raise ValueError(f"{symbol}: seq_len 与导出任务登记不一致")

        # baidu_keyword 本来就应该随股票不同，不纳入一致性比较。
        config = {
            k: v for k, v in prep["config"].items()
            if k != "baidu_keyword"
        }
        signature = {
            "config": config,
            "train_end": prep["train_end"],
            "val_end": prep["val_end"],
            "calendar": prep.get("calendar"),
            "version": prep["version"],
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"{symbol}: 时间切分/特征配置/日历不一致")

        bases = [c for c in columns if "__" not in c]
        for prefix in ("baidu_", "rzrq_"):
            if not any(c.startswith(prefix) for c in bases):
                raise ValueError(f"{symbol}: 缺少 {prefix} 数值特征")

        matrix = np.load(
            directory / "feature_matrix.npy", mmap_mode="r"
        )
        dates = np.load(directory / "dates.npy")
        if matrix.shape != (len(dates), len(columns)):
            raise ValueError(f"{symbol}: 矩阵维度不匹配")
        if not np.all(dates[1:] > dates[:-1]):
            raise ValueError(f"{symbol}: 日期未严格递增")
        del matrix

        sample = pd.read_csv(
            directory / "samples.csv",
            usecols=["split", "y_raw"],
        )
        train_y = sample.loc[sample.split == "train", "y_raw"]
        if train_y.empty:
            raise ValueError(f"{symbol}: 没有训练样本")

        records.append({
            "symbol": symbol,
            "root": str(directory),
            "prep": prep,
            "columns": columns,
            "dates": dates,
            "train_mean": float(train_y.mean()),
        })

    # 仅依据导出时在训练期保留的列生成 schema。
    exogenous = sorted({
        c
        for r in records
        for c in r["columns"]
        if c != TARGET and base_name(c) not in DROP_BASES
    })
    if not exogenous:
        raise ValueError("没有可用外生变量")

    return records, exogenous + [TARGET]


class PanelWindows(Dataset):
    """全局样本索引只定位到某股票内部，绝不跨股票切片。"""

    def __init__(self, records, schema, split):
        self.records = records
        self.schema = schema
        self.split = split
        self.parts = []
        self.cache = OrderedDict()
        self.seq_len = records[0]["prep"]["config"]["seq_len"]

        counts = []
        for r in records:
            frame = pd.read_csv(
                Path(r["root"]) / "samples.csv",
                usecols=[
                    "start_pos", "end_pos", "target_pos",
                    "split", "y_scaled", "y_raw",
                ],
            )
            frame = frame.loc[frame.split == split].copy()
            if frame.empty:
                raise ValueError(f"{r['symbol']}: {split} 没有样本")

            positions = frame[
                ["start_pos", "end_pos", "target_pos"]
            ].to_numpy(dtype=np.int64)

            start, end, target = positions.T
            if not (
                (start >= 0).all()
                and (end - start + 1 == self.seq_len).all()
                and (target == end + 1).all()
                and (target < len(r["dates"])).all()
            ):
                raise ValueError(f"{r['symbol']}: 窗口位置非法")

            target_dates = r["dates"][target].astype("datetime64[D]")
            train_end = np.datetime64(r["prep"]["train_end"], "D")
            val_end = np.datetime64(r["prep"]["val_end"], "D")

            if split == "train":
                valid = target_dates <= train_end
            elif split == "val":
                valid = (
                    (target_dates > train_end) & (target_dates <= val_end)
                )
            else:
                valid = target_dates > val_end
            if not valid.all():
                raise ValueError(f"{r['symbol']}: {split} 日期越界")

            stats = r["prep"]["statistics"][TARGET]
            expected = (
                frame.y_raw.to_numpy() - stats["mean"]
            ) / stats["scale"]
            ys = frame.y_scaled.to_numpy(dtype=np.float32)
            if not np.allclose(ys, expected, rtol=1e-5, atol=1e-5):
                raise ValueError(f"{r['symbol']}: 标签标准化不一致")

            self.parts.append((positions, ys))
            counts.append(len(frame))

        self.counts = np.asarray(counts, dtype=np.int64)
        self.ends = np.cumsum(self.counts)

        # 每个 epoch 所有股票总权重相同。
        self.stock_weights = (
            len(self) / (len(records) * self.counts)
        ).astype(np.float32)

        lookup = {c: i for i, c in enumerate(schema)}
        self.maps = []
        for r in records:
            pairs = [
                (lookup[c], i)
                for i, c in enumerate(r["columns"]) if c in lookup
            ]
            self.maps.append((
                np.array([p[0] for p in pairs], dtype=np.int64),
                np.array([p[1] for p in pairs], dtype=np.int64),
            ))
        self.missing = [
            i for i, c in enumerate(schema) if c.endswith("__missing")
        ]

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache"] = OrderedDict()
        return state

    def matrix(self, sid):
        if sid not in self.cache:
            path = Path(self.records[sid]["root"]) / "feature_matrix.npy"
            self.cache[sid] = np.load(path, mmap_mode="r")
            # 防止几千个股票矩阵长期同时占用文件描述符。
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        return self.cache[sid]

    def __getitem__(self, index):
        sid = bisect.bisect_right(self.ends, index)
        offset = 0 if sid == 0 else int(self.ends[sid - 1])
        local = index - offset

        positions, labels = self.parts[sid]
        start, end, target = positions[local]
        raw = self.matrix(sid)[start:end + 1]

        x = np.zeros(
            (self.seq_len, len(self.schema)), dtype=np.float32
        )
        x[:, self.missing] = 1.0
        dst, src = self.maps[sid]
        x[:, dst] = raw[:, src]

        y = np.array([[labels[local]]], dtype=np.float32)
        return x, y, sid, int(target)


class TypedExogenousEmbedding(nn.Module):
    """在官方外生 embedding 上增加变量身份编码。"""

    def __init__(self, original, feature_count, d_model):
        super().__init__()
        self.original = original
        self.feature_id = nn.Parameter(
            torch.empty(1, feature_count, d_model)
        )
        nn.init.normal_(self.feature_id, mean=0.0, std=0.02)

    def forward(self, x, x_mark=None):
        if x_mark is not None:
            raise ValueError("本实现不使用额外时间标记 token")
        embedded = self.original(x, None)
        if embedded.shape[1] != self.feature_id.shape[1]:
            raise ValueError("外生 token 数与 schema 不一致")
        return embedded + self.feature_id


def build_model(config):
    model = TimeXer(argparse.Namespace(**config)).float()
    model.ex_embedding = TypedExogenousEmbedding(
        model.ex_embedding,
        config["enc_in"] - 1,
        config["d_model"],
    )
    return model


def run_epoch(model, loader, device, optimizer=None):
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
        for x, y, sid, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sid = sid.to(device, non_blocking=True)

            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("输入/标签存在非有限值")

            pred = model(x, None, None, None)
            if pred.shape != y.shape:
                raise ValueError(f"输出 {pred.shape} != 标签 {y.shape}")

            per_sample = (pred - y).square().reshape(len(x), -1).mean(1)
            loss = (per_sample * weights[sid]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("loss 为 NaN/Inf")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.detach().item() * len(x)
            count += len(x)

    return total / count


@torch.no_grad()
def test_predictions(model, loader, records, device):
    model.eval()
    chunks = []

    for x, y, sid, target in loader:
        output = model(x.to(device), None, None, None)
        if output.shape != y.shape:
            raise ValueError("测试输出维度不正确")

        pred_scaled = output.cpu().numpy().reshape(-1)
        true_scaled = y.numpy().reshape(-1)
        last_scaled = x[:, -1, -1].numpy()

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
            "y_true": true_scaled * scales + means,
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


def metrics(frame, column):
    y = frame.y_true.to_numpy()
    pred = frame[column].to_numpy()
    error = pred - y
    nonzero = y != 0
    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "direction_accuracy_nonzero": (
            float((np.sign(pred[nonzero]) == np.sign(y[nonzero])).mean())
            if nonzero.any() else None
        ),
    }


def save_test_report(frame, out):
    predictors = [
        "y_pred", "baseline_zero", "baseline_mean",
        "baseline_persistence",
    ]
    per_stock = []
    summary = {}

    for column in predictors:
        stock_rows = []
        for symbol, part in frame.groupby("symbol"):
            row = {"symbol": symbol, "predictor": column}
            row.update(metrics(part, column))
            per_stock.append(row)
            stock_rows.append(row)

        daily_ic = []
        for _, part in frame.groupby("target_date"):
            if (
                len(part) >= 10
                and part.y_true.nunique() > 1
                and part[column].nunique() > 1
            ):
                value = part.y_true.rank().corr(part[column].rank())
                if np.isfinite(value):
                    daily_ic.append(float(value))

        summary[column] = {
            "pooled": metrics(frame, column),
            "stock_equal_mae": float(np.mean([
                row["mae"] for row in stock_rows
            ])),
            "stock_equal_mse": float(np.mean([
                row["mse"] for row in stock_rows
            ])),
            "mean_daily_rank_ic": (
                float(np.mean(daily_ic)) if daily_ic else None
            ),
            "rank_ic_days": len(daily_ic),
        }

    frame.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(per_stock).to_csv(
        out / "test_metrics_by_stock.csv", index=False
    )
    write_json(out / "test_metrics.json", summary)
    return summary


def train(args, device):
    if not args.data_root or not args.out_dir:
        raise ValueError("train 需要 --data-root 和 --out-dir")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pth"
    if checkpoint_path.exists():
        raise FileExistsError("输出目录已有 checkpoint，请换新目录")

    records, schema = load_catalog(args.data_root)
    train_ds = PanelWindows(records, schema, "train")
    val_ds = PanelWindows(records, schema, "val")

    seq_len = train_ds.seq_len
    if args.patch_len < 1 or seq_len % args.patch_len:
        raise ValueError("patch-len 必须整除 seq_len")
    if args.d_model % args.n_heads:
        raise ValueError("d-model 必须能被 n-heads 整除")

    config = {
        "task_name": "long_term_forecast",
        "features": "MS",
        "seq_len": seq_len,
        "pred_len": 1,
        "label_len": 0,
        "enc_in": len(schema),
        "c_out": 1,
        "d_model": args.d_model,
        "d_ff": args.d_model * 2,
        "e_layers": args.e_layers,
        "n_heads": args.n_heads,
        "factor": 1,
        "dropout": args.dropout,
        "embed": "fixed",
        "freq": "d",
        "activation": "gelu",
        # 保留已导出的尺度与 0/1 缺失标记，不再做窗口内归一化。
        "use_norm": 0,
        "patch_len": args.patch_len,
    }

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_ds, shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, **loader_options
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"stocks={len(records)}, features={len(schema)}, "
        f"train={len(train_ds)}, val={len(val_ds)}, device={device}"
    )

    metadata = {
        "format": "global_timexer_typed_v1",
        "model_config": config,
        "schema": schema,
        "preprocessing": {
            r["symbol"]: r["prep"] for r in records
        },
        "train_mean": {
            r["symbol"]: r["train_mean"] for r in records
        },
    }

    best = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_stock_equal_scaled_mse": train_loss,
            "val_stock_equal_scaled_mse": val_loss,
        })
        print(
            f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        write_json(out / "history.json", history)

        if val_loss < best:
            best = val_loss
            bad_epochs = 0
            state = {
                **metadata,
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "best_epoch": epoch,
                "best_val_loss": best,
            }
            temporary = out / "checkpoint.tmp"
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    # 训练结束、模型选定后，才执行测试预测。
    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(saved["state_dict"])

    test_ds = PanelWindows(records, schema, "test")
    test_loader = DataLoader(
        test_ds, shuffle=False, **loader_options
    )
    frame = test_predictions(model, test_loader, records, device)
    summary = save_test_report(frame, out)

    write_json(out / "schema.json", schema)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"共享 checkpoint 已保存：{checkpoint_path}")


@torch.no_grad()
def predict(args, device):
    if not args.checkpoint or not args.data_dir:
        raise ValueError("predict 需要 --checkpoint 和 --data-dir")

    saved = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if saved["format"] != "global_timexer_typed_v1":
        raise ValueError("不支持的 checkpoint 格式")

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

    model = build_model(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    z = model(
        torch.from_numpy(x).to(device), None, None, None
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--data-root")
    p.add_argument("--out-dir")
    p.add_argument("--data-dir")
    p.add_argument("--checkpoint")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    if min(
        args.epochs, args.patience, args.batch_size,
        args.d_model, args.e_layers, args.n_heads,
    ) < 1 or args.workers < 0:
        raise ValueError("数量参数不合法")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if args.mode == "train":
        train(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()