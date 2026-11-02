"""统一训练入口（模型无关）：train / refit 两个阶段的完整流程。

自 train_global_timexer.py 原样迁出（train / train_refit / CLI），
仅两处结构化改动，训练协议与产物契约不变：
1. 模型构造经适配器（--model 选择，默认 TimeXer）；
2. checkpoint 元数据新增 "model" 字段（适配器名），其余字段与
   旧产物逐字段一致（TimeXer 的 format 仍为 global_timexer_typed_v1）。

用法：
    python train_stock.py --model TimeXer --mode train --data-root ... --out-dir ...
    python train_stock.py --model TimeXer --mode refit ...
    python train_stock.py --mode predict --checkpoint ... --data-dir ...
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stock.adapters import get_adapter
from stock.data.panel import PanelWindows, load_catalog
from stock.data.schema import feature_schema_doc
from stock.inference.predictor import predict
from stock.training.artifacts import environment_info, prediction_time
from stock.training.trainer import run_epoch, test_predictions
from stock.evaluation.metrics import save_test_report
from stock.utils import now_text, read_json, sha256_file, write_json


def train(args, device):
    if not args.data_root or not args.out_dir:
        raise ValueError("train 需要 --data-root 和 --out-dir")

    adapter = get_adapter(args.model)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pth"
    if checkpoint_path.exists():
        raise FileExistsError("输出目录已有 checkpoint，请换新目录")

    # 阶段 1（选择）只允许 train_val_test 布局；train_test 重拟合导出
    # 请使用 --mode refit。
    records, schema = load_catalog(
        args.data_root, expected_layout="train_val_test"
    )
    train_ds = PanelWindows(records, schema, "train")
    # 单股验证覆盖不足（停牌/缺数）只影响该股早停贡献，不影响训练资格；
    # 整体验证集为空则无证据选预算，必须报错。
    val_ds = PanelWindows(records, schema, "val", allow_missing=True)
    if len(val_ds) == 0:
        raise ValueError(
            "整体验证集没有任何可评分样本，无法早停选预算"
        )

    seq_len = train_ds.seq_len
    if args.patch_len < 1 or seq_len % args.patch_len:
        raise ValueError("patch-len 必须整除 seq_len")
    if args.d_model % args.n_heads:
        raise ValueError("d-model 必须能被 n-heads 整除")

    config = adapter.model_config(args, seq_len, schema)

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

    model = adapter.build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"stocks={len(records)}, features={len(schema)}, "
        f"train={len(train_ds)}, val={len(val_ds)}, device={device}"
    )

    spec = read_json(Path(args.data_root) / "export_config.json")
    schema_info = feature_schema_doc(schema)

    metadata = {
        "format": adapter.checkpoint_format,
        "model": adapter.name,
        "model_config": config,
        "schema": schema,
        # 特征契约：完整列描述 + 稳定 hash；推理端按同一算法重算，
        # 不匹配即报错（旧产物缺字段时显式告警，见 predictor）。
        "feature_schema": schema_info,
        "feature_schema_hash": schema_info["hash"],
        # 数据集身份：导出配置字节哈希，便于预测记录回溯到数据口径。
        "dataset_id": sha256_file(Path(args.data_root) / "export_config.json"),
        "target_spec": records[0]["prep"].get("target_spec"),
        # 登记导出任务配置：本折数据截止日、训练起点、训练/验证边界、
        # 脚本与数据哈希，checkpoint 可直接追溯到生成它的数据口径
        # （walk-forward 每折不同）。
        "export_config": spec,
        "preprocessing": {
            r["symbol"]: r["prep"] for r in records
        },
        "train_mean": {
            r["symbol"]: r["train_mean"] for r in records
        },
    }

    best = float("inf")
    best_epoch = None
    bad_epochs = 0
    stopped_early = False
    history = []
    fit_started_at = now_text()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            adapter, model, train_loader, device, optimizer=optimizer,
            use_ex_mask=args.ex_mask,
        )
        val_loss = run_epoch(
            adapter, model, val_loader, device, use_ex_mask=args.ex_mask
        )
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
            best_epoch = epoch
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
            stopped_early = True
            break

    fit_finished_at = now_text()

    batches_per_epoch = len(train_loader)
    training_stats = {
        "n_train_samples": len(train_ds),
        "n_train_stocks": len(train_ds.record_ids),
        "n_val_samples": len(val_ds),
        "n_val_stocks": len(val_ds.record_ids),
        "batch_size": args.batch_size,
        "batches_per_epoch": int(batches_per_epoch),
        "completed_optimizer_steps": int(len(history) * batches_per_epoch),
        "best_epoch_optimizer_steps": int(best_epoch * batches_per_epoch),
    }

    # 选择阶段结果登记（先落盘再测试评估）：两阶段协议中 best_epoch 是
    # 重拟合阶段的训练预算（same_epochs 规则）；时点链用于检查
    # "数据截止 -> 拟合完成 -> 首个预测时刻"是否可部署。预算登记在
    # 测试评估之前完成，测试评估失败不丢预算记录。
    first_prediction = prediction_time(spec)
    selection_result = {
        "protocol": "selection_train_val_v1",
        "split_layout": "train_val_test",
        "as_of": spec.get("as_of"),
        "train_start": spec.get("train_start"),
        "train_end": spec.get("train_end"),
        "val_end": spec.get("val_end"),
        "test_start": spec.get("test_start"),
        "epochs_limit": args.epochs,
        "patience": args.patience,
        # 外生 token 掩码协议：refit 阶段必须与选择阶段一致（见
        # train_refit 的一致性检查），避免"选择带掩码、重拟合不带"
        # 的协议漂移。
        "ex_mask": bool(args.ex_mask),
        "completed_epochs": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_loss": best,
        # 重拟合训练预算规则：在更大的 Train+Val 上使用相同遍历次数
        # （相同 epoch 不等于相同更新步数，见 fit_metadata）。
        "budget_rule": "same_epochs",
        "training_stats": training_stats,
        "environment": environment_info(device),
        # 本阶段需要截至 val_end 的标签（训练 <= train_end、验证早停
        # <= val_end），必须赶在首个测试预测时刻之前完成。
        "fit_data_cutoff_time": f"{spec['val_end']}T15:00:00+08:00",
        "fit_data_cutoff_note": (
            "val_end 收盘行情与标签入库后（实际可用时间取决于采集延迟）"
        ),
        "fit_started_at": fit_started_at.isoformat(),
        "fit_finished_at": fit_finished_at.isoformat(),
        "first_prediction_time": first_prediction.isoformat(),
        "would_meet_first_prediction_deadline": bool(
            fit_finished_at <= first_prediction
        ),
    }
    write_json(out / "selection_result.json", selection_result)
    write_json(out / "schema.json", schema)
    write_json(out / "feature_schema.json", schema_info)

    # 训练结束、模型选定后，才执行测试预测（测试集按 allow_missing 装载，
    # 单股无测试样本只影响该股评估；整体为空说明测试块配置有问题）。
    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(saved["state_dict"])

    test_ds = PanelWindows(records, schema, "test", allow_missing=True)
    if len(test_ds) == 0:
        raise ValueError("整体测试集没有任何可评分样本")
    test_loader = DataLoader(
        test_ds, shuffle=False, **loader_options
    )
    frame = test_predictions(
        adapter, model, test_loader, records, device,
        use_ex_mask=args.ex_mask,
    )
    summary = save_test_report(frame, out)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"共享 checkpoint 已保存：{checkpoint_path}")


def train_refit(args, device):
    """两阶段协议阶段 2：Train+Val 重拟合。

    - 只接受 train_test 布局导出（Train+Val 已合并为训练集）；
    - 从头初始化，训练固定预算（--epochs，来自阶段 1 best_epoch），
      无验证集、无早停，完成预算后保存最终模型（不按训练损失挑选）；
    - 相同 epoch 数在扩大后的数据集上意味着每个 epoch 的 batch 数
      增加、优化器更新步数更多，这是 same_epochs 规则的已知语义；
    - 输出 fit_metadata.json 登记数据截止、拟合起止与首个预测时刻。
    """
    if not args.data_root or not args.out_dir:
        raise ValueError("refit 需要 --data-root 和 --out-dir")

    adapter = get_adapter(args.model)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pth"
    if checkpoint_path.exists():
        raise FileExistsError("输出目录已有 checkpoint，请换新目录")

    # 阶段 1 选择结果：best_epoch 即本阶段训练预算；提供时必须一致，
    # 防止预算与登记来源脱钩（如需偏离，请勿提供并自行登记）。
    selection = None
    if args.selection_result:
        selection = read_json(args.selection_result)
        if selection.get("protocol") != "selection_train_val_v1":
            raise ValueError(
                f"不支持的 selection_result 协议：{selection.get('protocol')}"
            )
        if int(selection["best_epoch"]) != args.epochs:
            raise ValueError(
                f"refit 训练预算必须等于选择阶段 best_epoch="
                f"{selection['best_epoch']}（当前 --epochs={args.epochs}）"
            )
        if selection.get("ex_mask", False) != bool(args.ex_mask):
            # 掩码协议必须前后一致：选择阶段与重拟合阶段的数据进入
            # 方式（哪些外生 token 参与注意力）不能改变。
            raise ValueError(
                f"refit 的外生掩码协议 --ex-mask={bool(args.ex_mask)} 与"
                f"选择阶段 ex_mask={selection.get('ex_mask', False)} 不一致"
            )

    records, schema = load_catalog(
        args.data_root, expected_layout="train_test"
    )
    train_ds = PanelWindows(records, schema, "train", layout="train_test")
    if len(train_ds) == 0:
        raise ValueError("重拟合训练集为空")

    # 选择阶段产物与重拟合导出的折边界必须一致：防止把别的折/别的
    # as-of 的 selection.json 当作本折预算来源（schema 因 enc_in 变化
    # 可以不同，但时间边界必须重合）。
    if selection:
        spec_check = read_json(Path(args.data_root) / "export_config.json")
        boundary_mismatch = []
        if selection.get("val_end") is not None and pd.Timestamp(
            selection["val_end"]
        ) != pd.Timestamp(spec_check["train_end"]):
            boundary_mismatch.append(
                f"selection.val_end={selection.get('val_end')} != "
                f"refit.train_end={spec_check.get('train_end')}"
            )
        if selection.get("as_of") is not None and str(
            selection["as_of"]
        ) != str(spec_check.get("as_of")):
            boundary_mismatch.append(
                f"selection.as_of={selection.get('as_of')} != "
                f"refit.as_of={spec_check.get('as_of')}"
            )
        if selection.get("test_start") is not None and pd.Timestamp(
            selection["test_start"]
        ) != pd.Timestamp(spec_check["test_start"]):
            boundary_mismatch.append(
                f"selection.test_start={selection.get('test_start')} != "
                f"refit.test_start={spec_check.get('test_start')}"
            )
        if boundary_mismatch:
            raise ValueError(
                "selection_result 与重拟合导出折边界不一致（"
                + "；".join(boundary_mismatch)
                + "）；请检查 --selection-json 是否指向本折的选择阶段产物"
            )

    seq_len = train_ds.seq_len
    if args.patch_len < 1 or seq_len % args.patch_len:
        raise ValueError("patch-len 必须整除 seq_len")
    if args.d_model % args.n_heads:
        raise ValueError("d-model 必须能被 n-heads 整除")

    config = adapter.model_config(args, seq_len, schema)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)

    spec = read_json(Path(args.data_root) / "export_config.json")
    schema_info = feature_schema_doc(schema)

    model = adapter.build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"[REFIT] stocks={len(records)}, features={len(schema)}, "
        f"train={len(train_ds)}, budget_epochs={args.epochs}, "
        f"device={device}"
    )

    metadata = {
        "format": adapter.checkpoint_format,
        "model": adapter.name,
        "model_config": config,
        "schema": schema,
        "feature_schema": schema_info,
        "feature_schema_hash": schema_info["hash"],
        "dataset_id": sha256_file(Path(args.data_root) / "export_config.json"),
        "target_spec": records[0]["prep"].get("target_spec"),
        "export_config": spec,
        "preprocessing": {
            r["symbol"]: r["prep"] for r in records
        },
        "train_mean": {
            r["symbol"]: r["train_mean"] for r in records
        },
        # 两阶段协议登记：predict 与 checkpoint 审计可直接识别本模型
        # 来自 Train+Val 重拟合，而非早停选择模型。
        "refit": {
            "protocol": "train_val_refit_v1",
            "fit_end": spec.get("train_end"),
            "training_budget": {
                "rule": "same_epochs", "epochs": args.epochs,
            },
            "initialization_type": "scratch",
            "selection_result": args.selection_result,
        },
    }

    fit_started_at = now_text()
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            adapter, model, train_loader, device, optimizer=optimizer,
            use_ex_mask=args.ex_mask,
        )
        history.append({
            "epoch": epoch,
            "train_stock_equal_scaled_mse": train_loss,
        })
        print(
            f"[REFIT] epoch={epoch}/{args.epochs} "
            f"train={train_loss:.6f}",
            flush=True,
        )
        write_json(out / "history.json", history)
    fit_finished_at = now_text()
    batches_per_epoch = len(train_loader)
    training_stats = {
        "n_train_samples": len(train_ds),
        "n_train_stocks": len(train_ds.record_ids),
        "batch_size": args.batch_size,
        "batches_per_epoch": int(batches_per_epoch),
        "completed_optimizer_steps": int(args.epochs * batches_per_epoch),
    }

    # 固定预算完成即保存最终模型：不按训练损失挑选、无早停。
    state = {
        **metadata,
        "state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
    }
    temporary = out / "checkpoint.tmp"
    torch.save(state, temporary)
    temporary.replace(checkpoint_path)

    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(saved["state_dict"])

    # 测试块内允许个别股票无有效样本（停牌等，覆盖率不决定训练资格），
    # 该股票不产生预测。整体测试集为空只发生在生产全量重拟合
    # （train_end == as_of，测试块尚无数据）：跳过测试评估、照常登记
    # fit_metadata；此时 walk-forward 汇总不适用（本模式不产生
    # test_predictions.csv）。
    test_ds = PanelWindows(
        records, schema, "test", layout="train_test", allow_missing=True
    )
    summary = None
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, shuffle=False, **loader_options)
        frame = test_predictions(
            adapter, model, test_loader, records, device,
            use_ex_mask=args.ex_mask,
        )
        summary = save_test_report(frame, out)
    else:
        print(
            "[REFIT] 测试区间没有任何可评分样本（生产全量重拟合 "
            "train_end == as-of），跳过测试评估"
        )

    write_json(out / "schema.json", schema)
    write_json(out / "feature_schema.json", schema_info)

    first_prediction = prediction_time(spec)
    fit_metadata = {
        "protocol": "train_val_refit_v1",
        "split_layout": "train_test",
        "as_of": spec.get("as_of"),
        "fit_start": spec.get("train_start"),
        "fit_end": spec.get("train_end"),
        "test_start": spec.get("test_start"),
        "refit_window_policy": args.refit_window_policy,
        "training_budget": {
            "rule": "same_epochs",
            "epochs": args.epochs,
            "selection_result": args.selection_result,
            "selection_best_epoch": (
                int(selection["best_epoch"]) if selection else None
            ),
            # 外生 token 掩码协议（与选择阶段一致性已在前面校验）。
            "ex_mask": bool(args.ex_mask),
            "selection_train_end": (
                selection.get("train_end") if selection else None
            ),
            "selection_val_end": (
                selection.get("val_end") if selection else None
            ),
            "note": (
                "same_epochs：验证阶段确定数据遍历次数，最终在更大的 "
                "Train+Val 上使用相同遍历次数；相同 epoch 意味着更多"
                "优化器更新步数（每个 epoch 的 batch 数增加）"
            ),
        },
        "initialization_type": "scratch",
        # warm-start 预留字段：当前协议每折从头训练，无父模型。
        "parent_checkpoint": None,
        "parent_label_cutoff": None,
        "training_stats": training_stats,
        "environment": environment_info(device),
        # 测试评估是否完成（生产全量重拟合 train_end == as-of 时为 False）。
        "test_evaluated": summary is not None,
        "universe_decision_time": args.universe_decision_time,
        "universe_snapshot_hash": spec.get("universe_sha256"),
        "schema_hash": hashlib.sha256(
            json.dumps(schema, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "data_snapshot": {
            "db": spec.get("db"),
            "dataset_script_sha256": spec.get("dataset_script_sha256"),
            "export_script_sha256": spec.get("export_script_sha256"),
        },
        # 时点链：fit_data_cutoff_time（fit_end 收盘标签入库后）
        # <= fit_started_at < fit_finished_at <= first_prediction_time。
        "fit_data_cutoff_time": f"{spec['train_end']}T15:00:00+08:00",
        "fit_data_cutoff_note": (
            "fit_end（= 选择阶段 val_end）收盘行情与标签入库后才开始"
            "拟合；实际可用时间取决于采集延迟，离线回测不能证明当时"
            "能及时完成"
        ),
        "fit_started_at": fit_started_at.isoformat(),
        "fit_finished_at": fit_finished_at.isoformat(),
        "first_prediction_time": first_prediction.isoformat(),
        "would_meet_first_prediction_deadline": bool(
            fit_finished_at <= first_prediction
        ),
        "warm_start": {
            "implemented": False,
            "note": (
                "跨折 warm-start 未实现，保留为独立实验；两个前提问题："
                "1) 每折重新拟合每股 scaler/schema/特征保留列后，上一折"
                "权重不能当作无条件兼容的连续训练状态；2) 上一折最终"
                "模型可能已见过本折验证窗口标签，不能直接用于本折早停选择"
            ),
        },
    }
    write_json(out / "fit_metadata.json", fit_metadata)

    if summary is not None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"重拟合 checkpoint 已保存：{checkpoint_path}")


def build_train_parser():
    p = argparse.ArgumentParser(
        description="股票统一训练入口（模型经适配器接入，默认 TimeXer）"
    )
    p.add_argument(
        "--model", default="TimeXer",
        help="模型适配器名（见 stock/adapters/，默认 TimeXer）",
    )
    p.add_argument(
        "--mode", choices=["train", "predict", "refit"], default="train",
        help="train=阶段 1（Val 早停选择）；refit=阶段 2（Train+Val "
             "重拟合，固定预算、无早停）；predict=推理",
    )
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
    p.add_argument(
        "--ex-mask",
        action="store_true",
        help="屏蔽窗口内整段无有效观测的外生值 token（判定只看"
             "值列的 __missing 标记；缺失标记/年龄列本身不屏蔽）。"
             "默认关闭＝维持既有\"填补＋缺失标记\"协议；数据集恒产出"
             "掩码，本开关仅控制是否传入模型。启用后登记进 "
             "checkpoint，predict / predict_daily 自动跟随；建议按"
             "预先规定的消融实验（开 vs 关）对比后再采用",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--selection-result",
        help="refit：阶段 1 的 selection_result.json；提供时 best_epoch "
             "必须等于 --epochs（训练预算与登记来源绑定）",
    )
    p.add_argument(
        "--refit-window-policy",
        choices=["extend_to_val_end", "keep_train_days"],
        default="extend_to_val_end",
        help="refit：重拟合训练窗口策略（登记到 fit_metadata.json；"
             "窗口本身由导出的 --train-start 决定，此处仅登记语义）",
    )
    p.add_argument(
        "--universe-decision-time",
        help="refit：股票池资格决定时点（PIT 模式由调度器传入；登记用）",
    )
    return p


def main(argv=None):
    args = build_train_parser().parse_args(argv)

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
    elif args.mode == "refit":
        train_refit(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
