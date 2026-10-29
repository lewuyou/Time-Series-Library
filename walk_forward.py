# -*- coding: utf-8 -*-
"""最小 walk-forward 滚动评估调度器（完整流程第 5 步，可选）。

协议（尽量复用现有导出与训练脚本，不改动其核心逻辑）：
    每折：独立导出 → 独立拟合预处理（特征筛选/填补/scaler）→
          从头初始化模型 → Train 训练 → Val 早停选 checkpoint →
          对紧随其后的 Test 块逐日一步预测。

    - 测试块互不重叠，滚动步长等于测试块长度（--test-days）；
    - expanding（默认）：训练目标日为日历起点至本折 train-end 的全部
      可用历史；rolling（--train-days N）：限制为最近 N 个候选训练
      目标交易日（不保证每只股票有 N 个有效样本，单股下限由
      --min-train-samples 兜底）；
    - 尾部不足一个完整测试块的日期直接跳过（第一版不做部分折）；
    - 测试块内模型参数固定，后面的测试日可以使用此前已经发生的
      测试期行情作为输入（与 PanelWindows 行为一致），不是在测试块
      开始时一次性预测整个未来；
    - 单折导出的断点恢复（export_config.json 一致性复检）仍然保留；
      调度层不实现断点恢复，中断后请换新的 --run-root 重跑。

日期边界（全市场统一交易日历 D0..Dn）：
    第 k 折第一个测试日位置 b = first_test_pos + k * test_days，
    train_end_pos = b - val_days - 1，val_end_pos = b - 1，
    test_end_pos = b + test_days - 1（即本折 as-of）。

数据库快照约束：
    运行期间数据库必须保持不变（建议固定快照，不要让采集任务继续
    更新同一份库）。每折导出前校验最新行情日与截至该日的交易日历
    指纹，与启动时不一致即硬失败，避免前后折混用不同版本的历史。
    该检查不能发现不改变日期的历史数值修订，完整保证仍需固定快照
    （配置和脚本哈希不能替代数据快照）。

输出（--run-root）：
    walk_forward_plan.json     协议、参数、折边界、数据库与脚本指纹
    folds.csv                  逐折日期边界表
    fold_metrics.json          各折 test_metrics.json 汇总
    oos_summary.json           拼接后样本外总指标（原始收益率尺度）
    data/<fold_id>/            每折导出目录（export_global.py 输出）
    checkpoints/<fold_id>/     每折训练输出（train_global_timexer.py）
    oos/test_predictions.csv   所有折样本外预测合集，附 fold_id 与该折
                               train_start/train_end/val_end，可追溯每条
                               预测来自哪个训练版本
    oos/test_metrics.json      拼接后总指标
    oos/test_metrics_by_stock.csv

已知局限（在单次评估基础上新增/放大，详见 train_global_timexer.py）：
    1. 固定输入 universe.csv 属于条件性历史评估：股票池由当前市值、
       当前存续名单、当前简称等决定，walk-forward 不能解决股票池的
       point-in-time 问题；
    2. 每折模型参数截至 train_end，Val 仅用于选择 epoch，与"每月用
       截至昨天的全部标签重训"的部署方式不同；
    3. 每折实际入模股票可能因训练期辅助数据覆盖不同而变化
       （filtered_no_baidu / filtered_no_rzrq），差异按折记录在
       data/<fold_id>/export_results.csv；不要事后取所有折都成功的
       股票交集重训，否则重新引入未来筛选；
    4. 评估仍是统计口径收益率，不是可实现交易收益；
    5. --min-eval-samples >= --test-days 时要求每只股票测试块内全部
       交易日有效（例如 20/20），约束严格；可事先统一选择较低下限
       （如 10），但不能看完某折结果后再有针对性地调整。

运行示例：
    # expanding
    python walk_forward.py \
        --db ../AutoTS/download/autots.duckdb \
        --universe-csv dataset/global_001/universe.csv \
        --run-root runs/wf_expanding_001 \
        --first-test-date 2024-01-02 --last-test-date 2024-12-31 \
        --val-days 60 --test-days 20 --seq-len 60 \
        --min-train-samples 250 --min-eval-samples 20 \
        --export-workers 4 --epochs 30 --patience 5 \
        --batch-size 256 --loader-workers 0 \
        --d-model 64 --e-layers 2 --patch-len 10 --device auto

    # rolling 对照（最近约三年训练标签）：增加 --train-days 732，
    # 并使用新的 --run-root runs/wf_rolling_732_001
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

import dataset as ds
import train_global_timexer

SCRIPT_DIR = Path(__file__).resolve().parent


def date_text(value):
    return str(pd.Timestamp(value).date())


def read_db_state(con, calendar):
    """数据库快照指纹：最新行情日 + 截至该日的交易日历哈希。

    能发现追加新行情/新交易日；不能发现不改变日期的历史数值修订，
    完整保证需使用固定数据库快照。
    """
    value = con.execute(
        "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
    ).fetchone()[0]
    if value is None:
        raise ValueError("数据库没有后复权行情")
    latest = pd.Timestamp(value).normalize()
    sha, days = ds.calendar_dates_sha256(calendar, latest)
    return {
        "latest_quote_date": str(latest.date()),
        "calendar_sha256_to_latest": sha,
        "trading_days_to_latest": int(days),
    }


def build_folds(calendar, first_test_date, last_test_date, val_days,
                test_days, train_days=None):
    """仅生成完整测试块；train_days=None 表示 expanding。"""
    calendar = pd.DatetimeIndex(calendar).normalize()
    first = pd.Timestamp(first_test_date).normalize()
    last = pd.Timestamp(last_test_date).normalize()

    if first not in calendar or last not in calendar:
        raise ValueError("first-test-date/last-test-date 必须是交易日")
    if first > last:
        raise ValueError("first-test-date 不能晚于 last-test-date")

    first_pos = calendar.get_loc(first)
    last_pos = calendar.get_loc(last)

    folds = []
    begin = first_pos

    while begin + test_days - 1 <= last_pos:
        test_end_pos = begin + test_days - 1
        val_end_pos = begin - 1
        train_end_pos = begin - val_days - 1

        if train_end_pos < 0:
            raise ValueError("首折验证集之前没有足够训练历史")

        train_start = None
        if train_days is not None:
            train_start_pos = train_end_pos - train_days + 1
            if train_start_pos < 0:
                raise ValueError("首折不足指定 rolling 训练长度")
            train_start = date_text(calendar[train_start_pos])

        folds.append({
            "fold_id": f"fold_{len(folds):03d}",
            "train_start": train_start,
            "train_end": date_text(calendar[train_end_pos]),
            "val_end": date_text(calendar[val_end_pos]),
            "test_start": date_text(calendar[begin]),
            "as_of": date_text(calendar[test_end_pos]),
            "val_days": val_days,
            "test_days": test_days,
        })

        begin += test_days

    if not folds:
        raise ValueError("指定区间不足一个完整测试块")

    return folds


def run_command(command):
    print("\n>>>", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--db", default=ds.DEFAULT_DB)
    p.add_argument("--universe-csv", required=True)
    p.add_argument("--run-root", required=True)

    p.add_argument("--first-test-date", required=True)
    p.add_argument("--last-test-date", required=True)

    p.add_argument("--val-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument(
        "--train-days", type=int,
        help="rolling 训练目标交易日数；不传为 expanding",
    )

    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--min-train-samples", type=int, default=250)
    p.add_argument(
        "--min-eval-samples", type=int, default=20,
        help="单股每折 val/test 有效样本下限；等于 test-days 时要求测试块"
             "内全部交易日有效，可事先统一选择较低下限（如 10），但不能"
             "看完某折结果后再调整",
    )
    p.add_argument("--export-workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--loader-workers", type=int, default=0)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--patch-len", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
    )
    return p.parse_args()


def main():
    a = parse_args()

    if min(
        a.val_days, a.test_days, a.seq_len,
        a.min_train_samples, a.min_eval_samples, a.export_workers,
    ) < 1:
        raise ValueError("数量参数必须为正数")

    if a.train_days is not None and a.train_days < 1:
        raise ValueError("train-days 必须为正数")

    if min(a.val_days, a.test_days) < a.min_eval_samples:
        raise ValueError(
            "当前保留单股评估样本下限，"
            "val-days/test-days 不能小于 min-eval-samples"
        )

    db = str(Path(a.db).resolve())
    universe = str(Path(a.universe_csv).resolve())
    root = Path(a.run_root).resolve()

    if root.exists():
        raise FileExistsError(
            f"{root} 已存在；本调度器不实现断点恢复，请使用新的 run-root"
        )
    if not Path(db).exists():
        raise FileNotFoundError(f"数据库不存在：{db}")
    if not Path(universe).exists():
        raise FileNotFoundError(f"股票池文件不存在：{universe}")

    con = duckdb.connect(db, read_only=True)
    try:
        calendar, _ = ds.load_calendar(
            con, None, ds.DEFAULT_CALENDAR_START, 50
        )
        baseline_state = read_db_state(con, calendar)
        latest = pd.Timestamp(baseline_state["latest_quote_date"])

        last_test = pd.Timestamp(a.last_test_date).normalize()
        if last_test > latest:
            raise ValueError("last-test-date 晚于库内最新行情")

        folds = build_folds(
            calendar=calendar,
            first_test_date=a.first_test_date,
            last_test_date=a.last_test_date,
            val_days=a.val_days,
            test_days=a.test_days,
            train_days=a.train_days,
        )

        last_asof = pd.Timestamp(folds[-1]["as_of"])
        if calendar.get_loc(last_asof) + 1 >= len(calendar):
            raise ValueError("日历缺少最后一折 as-of 之后的下一交易日")

        root.mkdir(parents=True)
        (root / "data").mkdir()
        (root / "checkpoints").mkdir()
        (root / "oos").mkdir()

        plan = {
            "protocol": "independent_train_val_test_v1",
            "training_window": (
                "rolling" if a.train_days is not None else "expanding"
            ),
            "args": vars(a),
            "db": db,
            "db_state": baseline_state,
            "universe_csv": universe,
            "universe_sha256": ds.sha256_file(universe),
            "dataset_script_sha256": ds.sha256_file(ds.__file__),
            "export_script_sha256": ds.sha256_file(
                SCRIPT_DIR / "export_global.py"
            ),
            "train_script_sha256": ds.sha256_file(
                SCRIPT_DIR / "train_global_timexer.py"
            ),
            "folds": folds,
            "actual_last_test_date": folds[-1]["as_of"],
            "limitations": [
                "固定输入股票池属于条件性历史评估，不是 point-in-time "
                "股票池回测",
                "每折参数截至 train_end，Val 仅用于早停选 checkpoint，"
                "与部署期滚动重训口径不同",
                "每折实际入模股票可能不同（训练期辅助数据覆盖筛选），"
                "按折报告，不做事后交集重训",
                "统计口径收益率评估，不是可实现交易收益",
            ],
        }
        ds.json_write(root / "walk_forward_plan.json", plan)
        pd.DataFrame(folds).to_csv(root / "folds.csv", index=False)

        if folds[-1]["as_of"] != date_text(last_test):
            print(
                "[INFO] 尾部不足完整测试块的日期已跳过；"
                f"实际测试结束日={folds[-1]['as_of']}"
            )

        predictions = []
        fold_metrics = {}

        for fold in folds:
            fold_id = fold["fold_id"]

            # 数据库快照守护：追加行情或日历变化即硬失败。
            current_state = read_db_state(con, calendar)
            if current_state != baseline_state:
                raise ValueError(
                    "数据库在运行期间发生变化（最新行情日或交易日历指纹"
                    "不一致）；请使用固定数据库快照，并换新的 run-root 重跑"
                )

            data_root = root / "data" / fold_id
            out_dir = root / "checkpoints" / fold_id

            export_command = [
                sys.executable,
                str(SCRIPT_DIR / "export_global.py"),
                "--db", db,
                "--universe-csv", universe,
                "--data-root", str(data_root),
                "--workers", str(a.export_workers),
                "--seq-len", str(a.seq_len),
                "--test-days", str(a.test_days),
                "--val-days", str(a.val_days),
                "--min-train-samples", str(a.min_train_samples),
                "--min-eval-samples", str(a.min_eval_samples),
                "--as-of", fold["as_of"],
                "--train-end", fold["train_end"],
                "--val-end", fold["val_end"],
            ]
            if fold["train_start"] is not None:
                export_command += ["--train-start", fold["train_start"]]

            run_command(export_command)

            # 折边界登记复核：导出登记必须与调度计划完全一致。
            exported = json.loads(
                (data_root / "export_config.json").read_text(
                    encoding="utf-8"
                )
            )
            for key in ("as_of", "train_end", "val_end", "train_start"):
                if exported.get(key) != fold[key]:
                    raise ValueError(
                        f"{fold_id}: 导出登记 {key}={exported.get(key)} "
                        f"与计划 {fold[key]} 不一致"
                    )

            train_command = [
                sys.executable,
                str(SCRIPT_DIR / "train_global_timexer.py"),
                "--mode", "train",
                "--data-root", str(data_root),
                "--out-dir", str(out_dir),
                "--epochs", str(a.epochs),
                "--patience", str(a.patience),
                "--batch-size", str(a.batch_size),
                "--workers", str(a.loader_workers),
                "--d-model", str(a.d_model),
                "--e-layers", str(a.e_layers),
                "--n-heads", str(a.n_heads),
                "--patch-len", str(a.patch_len),
                "--lr", str(a.lr),
                "--dropout", str(a.dropout),
                "--weight-decay", str(a.weight_decay),
                "--seed", str(a.seed),
                "--device", a.device,
            ]
            run_command(train_command)

            frame = pd.read_csv(
                out_dir / "test_predictions.csv",
                dtype={"symbol": str},
            )
            target_dates = pd.to_datetime(frame["target_date"])

            inside = (
                target_dates.ge(pd.Timestamp(fold["test_start"]))
                & target_dates.le(pd.Timestamp(fold["as_of"]))
            )
            if frame.empty or not inside.all():
                raise ValueError(f"{fold_id}: 测试预测为空或日期越界")

            if frame.duplicated(["symbol", "target_date"]).any():
                raise ValueError(f"{fold_id}: 单折预测存在重复")

            frame["fold_id"] = fold_id
            frame["train_start"] = fold["train_start"]
            frame["train_end"] = fold["train_end"]
            frame["val_end"] = fold["val_end"]
            predictions.append(frame)

            fold_metrics[fold_id] = json.loads(
                (out_dir / "test_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            print(
                f"\n[FOLD OK] {fold_id}: rows={len(frame)}, "
                f"stocks={frame['symbol'].nunique()}, "
                f"test={fold['test_start']}~{fold['as_of']}"
            )

        oos = pd.concat(predictions, ignore_index=True)
        oos = oos.sort_values(
            ["target_date", "symbol"]
        ).reset_index(drop=True)

        if oos.duplicated(["symbol", "target_date"]).any():
            raise ValueError("不同 fold 的 OOS 预测存在重复")

        # 复用训练脚本的报告函数：各折预测已按该折 scaler 还原到原始
        # 收益率尺度，拼接后统一评估；不直接平均各折标准化指标。
        summary = train_global_timexer.save_test_report(oos, root / "oos")

        ds.json_write(root / "fold_metrics.json", fold_metrics)
        ds.json_write(root / "oos_summary.json", summary)

        print(
            f"\n[OK] folds={len(folds)}, oos_rows={len(oos)}, "
            f"stocks={oos['symbol'].nunique()}, "
            f"dates={oos['target_date'].nunique()}"
        )
        print(f"输出目录：{root}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
