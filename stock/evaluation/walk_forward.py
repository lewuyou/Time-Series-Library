# -*- coding: utf-8 -*-
"""最小 walk-forward 滚动评估调度器（完整流程第 5 步，可选）。

协议（尽量复用现有导出与训练脚本，不改动其核心逻辑）：

[默认：无重拟合协议 A，independent_train_val_test_v1]
    每折：独立导出 → 独立拟合预处理（特征筛选/填补/scaler）→
          从头初始化模型 → Train 训练 → Val 早停选 checkpoint →
          对紧随其后的 Test 块逐日一步预测。

[--refit：两阶段协议 B，train_val_refit_v1，推荐对照实验]
    每折拆成"选择"与"重拟合"两个阶段（设训练截止日 A、验证截止日
    B、测试截止日 C）：
    阶段 1 选择：Train <= A 训练、A~B 验证早停选出 best_epoch
        （train_val_test 导出、独立预处理 P_select）；
    阶段 2 重拟合：train_test 导出（fit_end=B，含原验证区间标签）、
        独立重新拟合预处理 P_refit（Train+Val 上重算特征覆盖率/保留列/
        填补/scaler/目标标准化/schema）、从头初始化、训练固定
        best_epoch 个 epoch（same_epochs 预算规则）、无验证集、无早停、
        完成预算后保存最终模型；
    阶段 3 样本外：测试块 B~C 由重拟合模型逐日预测（参数冻结，测试块
        内可使用已发生的测试期行情作为输入）。
    验证集可以进入最终训练，因为它在测试开始前已是历史；测试集仍不
    参与任何训练期决策。选择阶段的测试预测同时保留为协议 A 对照
    （oos/selection_control/），两协议使用完全相同的 OOS 日期。
    same_epochs 语义：相同 epoch 数在扩大后的数据集上意味着每个
    epoch 的 batch 数增加、优化器更新步数更多；不用后面折的结果调整
    前面折的训练预算。跨折 warm-start 不在本协议内（见"已知局限"）。

[--pit-universe：每折部分 PIT 股票池]
    每折调用 `python -m stock.selection.universe --as-of` 独立生成股票池：
    选择阶段按 A 时点、重拟合阶段按 B 时点重新确定资格（U_select 与
    U_refit 可不同）。可按时点评估的资格项：上市日期、行情历史、两融/百度
    历史；市值排名与简称映射仍是当前快照（部分 PIT，详见
    stock/selection/universe.py）。
    测试块内不新增股票；按当时交易状态与输入有效性决定是否发信号。
    不要根据测试期表现或数据完整性回头筛股票。

通用约束：
    - 测试块互不重叠，滚动步长等于测试块长度（--test-days）；
    - expanding（默认）：训练目标日为日历起点至本折 train-end 的全部
      可用历史；rolling（--train-days N）：限制为最近 N 个候选训练
      目标交易日（不保证每只股票有 N 个有效样本，单股下限由
      --min-train-samples 兜底）；
    - --refit-window-policy：extend_to_val_end（默认，训练起点不变、
      终点延伸至 B，即真正的 Train+Val 合并）；keep_train_days（终点
      延伸至 B、起点同步后移，保持最近 --train-days 个训练日；需要
      --train-days）；
    - 尾部不足一个完整测试块的日期直接跳过（第一版不做部分折）；
    - 测试块内模型参数固定（选择阶段用 best checkpoint、重拟合阶段
      用完成预算后的最终 checkpoint），后面的测试日可以使用此前已经
      发生的测试期行情作为输入（与 PanelWindows 行为一致），不是在
      测试块开始时一次性预测整个未来；
    - 单折导出的断点恢复（export_config.json 一致性复检）仍然保留；
      调度层不实现断点恢复，中断后请换新的 --run-root 重跑。

日期边界（全市场统一交易日历 D0..Dn）：
    第 k 折第一个测试日位置 b = first_test_pos + k * test_days，
    train_end_pos = b - val_days - 1，val_end_pos = b - 1，
    test_end_pos = b + test_days - 1（即本折 as-of）。

数据库快照约束：
    运行期间数据库必须保持不变（建议固定快照，不要让采集任务继续
    更新同一份库）。每折各阶段导出前校验最新行情日与截至该日的交易日
    历指纹，与启动时不一致即硬失败，避免前后折混用不同版本的历史。
    该检查不能发现不改变日期的历史数值修订，完整保证仍需固定快照
    （配置和脚本哈希不能替代数据快照）。

输出（--run-root）：
    walk_forward_plan.json     协议、参数、折边界、数据库与脚本指纹
    folds.csv                  逐折日期边界表（--refit 时含 refit_train_start）
    fold_metrics.json          各折 test_metrics.json 汇总（--refit 时
                               分 selection/refit 两阶段嵌套）
    oos_summary.json           拼接后样本外总指标（原始收益率尺度）；
                               --refit 时含重拟合/选择对照/公共样本诊断
    fit_timing.csv             （--refit）各阶段"数据截止 -> 拟合起止 ->
                               首个预测时刻"时点链登记
    data/<fold_id>/            每折导出目录（stock.data.export 输出；
                               --refit 时分 selection/ 与 refit/ 两套）
    data/<fold_id>/pit_universes.json（--pit-universe）每折股票池资格
                               时点、路径与哈希
    checkpoints/<fold_id>/     每折训练输出（--refit 时分 selection/ 与
                               refit/：checkpoint、selection_result.json、
                               fit_metadata.json）
    oos/test_predictions.csv   所有折样本外预测合集（--refit 时来自重拟合
                               模型），附 fold_id 与该折训练边界，可追溯
                               每条预测来自哪个训练版本
    oos/selection_daily.csv    逐日选股胜率（fold_id/信号数/命中数/胜率/
                               同日基线上涨率/oracle 上限）
    oos/selection_summary.json 整体与逐折选股胜率、跨折 mean/worst 汇总
    oos/selection_control/     （--refit）协议 A 对照：选择阶段测试预测
                               与指标（含同名 selection_daily.csv /
                               selection_summary.json）
    oos/comparison.json         并入 oos_summary.json（--refit）
    oos/test_metrics.json      拼接后总指标
    oos/test_metrics_by_stock.csv

已知局限（在单次评估基础上新增/放大，详见 stock/training/runner.py）：
    1. 固定输入 universe.csv 属于条件性历史评估：股票池由当前市值、
       当前存续名单、当前简称等决定，walk-forward 不能解决股票池的
       point-in-time 问题；--pit-universe 只是部分 PIT（市值排名与
       简称映射仍是当前快照），历史市值/简称补齐前不应视为真正 PIT；
    2. 无重拟合协议的参数截至 train_end，Val 仅用于选 epoch；重拟合
       协议通过 Train+Val refit 缓解，但 same_epochs 预算下相同 epoch
       意味着更多优化器更新步数（已知语义，不是等量更新）；
    3. fit_timing 的 would_meet_first_prediction_deadline 只是离线
       回测的墙上时钟记录，不能证明当时能按时完成（采集延迟、算力
       差异都未建模）；
    4. 每折实际入模股票可能因训练期辅助数据覆盖不同而变化
       （filtered_no_baidu / filtered_no_rzrq），差异按折记录在
       data/<fold_id>/export_results.csv；不要事后取所有折都成功的
       股票交集重训，否则重新引入未来筛选；
    5. 评估仍是统计口径收益率，不是可实现交易收益；
    6. 单股验证/测试覆盖率只登记不强制、不决定训练资格（两种布局
       一致）：评估期停牌/缺数/退市不影响训练资格，避免测试期筛选
       偏差；--min-eval-samples 仅作为登记与配置一致性检查（评估块
       长度对照），整体验证集/测试集为空才会在训练装载时报错；
    7. 跨折 warm-start 未实现，保留为独立实验：1) 每折重新拟合每股
       scaler/schema/特征保留列后，上一折权重不能当作无条件兼容的
       连续训练状态；2) 上一折最终模型可能已见过本折验证窗口标签，
       不能直接用于本折早停选择。

运行示例：
    # expanding，无重拟合（协议 A）
    python -m stock.evaluation.walk_forward \
        --db ../AutoTS/download/autots.duckdb \
        --universe-csv dataset/global_001/universe.csv \
        --run-root runs/wf_expanding_001 \
        --first-test-date 2024-01-02 --last-test-date 2024-12-31 \
        --val-days 60 --test-days 20 --seq-len 60 \
        --min-train-samples 250 --min-eval-samples 20 \
        --export-workers 4 --epochs 30 --patience 5 \
        --batch-size 256 --loader-workers 0 \
        --d-model 64 --e-layers 2 --patch-len 10 --device auto

    # 两阶段协议（协议 B），rolling 对照（最近约三年训练标签）：
    # 增加 --train-days 732 --refit，并使用新的 --run-root
    python -m stock.evaluation.walk_forward \
        --db ../AutoTS/download/autots.duckdb \
        --universe-csv dataset/global_001/universe.csv \
        --run-root runs/wf_refit_rolling_732_001 \
        --first-test-date 2024-01-02 --last-test-date 2024-12-31 \
        --val-days 60 --test-days 20 --seq-len 60 --train-days 732 \
        --refit --refit-window-policy extend_to_val_end \
        --min-train-samples 250 --min-eval-samples 20 \
        --export-workers 4 --epochs 30 --patience 5 \
        --batch-size 256 --loader-workers 0 --device auto

    # 每折部分 PIT 股票池（选择按 A 时点、重拟合按 B 时点）：
    # 用 --pit-universe 替代 --universe-csv，并可调 --pit-top-mv 等
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import dataset as ds

from stock.adapters import get_adapter
from stock.data import export as export_module
from stock.evaluation.daily_hit import daily_hit
from stock.evaluation.metrics import save_test_report
from stock.selection import universe as universe_module
from stock.training import runner as runner_module
from stock.utils import date_text

# stock/evaluation/walk_forward.py -> 仓库根（子进程用 -m 调用 stock.* 模块）
SCRIPT_DIR = Path(__file__).resolve().parents[2]

# runner 侧的 BooleanOptionalAction 参数：值为 False 时必须显式传
# --no-*（不能省略，否则会落到 runner 的默认值）。
_BOOLOPT_PARAMS = {"head_calendar"}


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


def ensure_db_state(con, baseline_state, stage_label):
    """各阶段导出前的数据库快照守护：追加行情或日历变化即硬失败。

    指纹计算使用的日历在每次检查时从数据库重新推导（而不是复用
    进程启动时的加载结果），这样"日历表变化但行情最新日期未变"
    的情况（如 holiday_calendar 补录休市日）也能被发现。
    """
    current_calendar, _ = ds.load_calendar(
        con, None, ds.DEFAULT_CALENDAR_START, 50
    )
    current_state = read_db_state(con, current_calendar)
    if current_state != baseline_state:
        raise ValueError(
            f"数据库在运行期间发生变化（{stage_label}：最新行情日或"
            "交易日历指纹不一致）；请使用固定数据库快照，并换新的 "
            "run-root 重跑"
        )


def build_folds(calendar, first_test_date, last_test_date, val_days,
                test_days, train_days=None, refit=False,
                refit_window_policy="extend_to_val_end"):
    """仅生成完整测试块；train_days=None 表示 expanding。

    refit=True 时同时给出每折重拟合训练窗口起点 refit_train_start：
      extend_to_val_end：与选择阶段训练起点一致，终点延伸至 B
        （真正的 Train+Val 合并；expanding 时仍为 None）；
      keep_train_days：终点为 B、起点后移，保持最近 train_days 个
        训练目标日（"用验证定预算，再用截至 B 的最近 N 日训练"）。
    """
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

        fold = {
            "fold_id": f"fold_{len(folds):03d}",
            "train_start": train_start,
            "train_end": date_text(calendar[train_end_pos]),
            "val_end": date_text(calendar[val_end_pos]),
            "test_start": date_text(calendar[begin]),
            "as_of": date_text(calendar[test_end_pos]),
            "val_days": val_days,
            "test_days": test_days,
        }

        if refit:
            if refit_window_policy == "keep_train_days":
                if train_days is None:
                    raise ValueError(
                        "refit-window-policy=keep_train_days 需要 --train-days"
                    )
                refit_start_pos = val_end_pos - train_days + 1
                if refit_start_pos < 0:
                    raise ValueError("首折不足指定 rolling 重拟合训练长度")
                fold["refit_train_start"] = date_text(
                    calendar[refit_start_pos]
                )
            else:
                # extend_to_val_end：起点与选择阶段一致，终点延伸至 B。
                fold["refit_train_start"] = train_start

        folds.append(fold)

        begin += test_days

    if not folds:
        raise ValueError("指定区间不足一个完整测试块")

    return folds


def run_command(command):
    print("\n>>>", " ".join(map(str, command)), flush=True)
    # 子进程用 python -m 调 stock.* 模块：保留调用方 cwd（相对路径参数
    # 语义不变），只把仓库根加进 PYTHONPATH 保证模块可导入。
    env = os.environ.copy()
    root = str(SCRIPT_DIR)
    env["PYTHONPATH"] = (
        root if not env.get("PYTHONPATH")
        else root + os.pathsep + env["PYTHONPATH"]
    )
    subprocess.run(command, check=True, env=env)


def export_command(a, db, universe, data_root, fold, layout,
                  train_end, val_end=None, train_start_key="train_start"):
    """构造 stock.data.export 命令（显式日期边界）。"""
    command = [
        sys.executable,
        "-m", "stock.data.export",
        "--db", db,
        "--universe-csv", str(universe),
        "--data-root", str(data_root),
        "--workers", str(a.export_workers),
        "--seq-len", str(a.seq_len),
        "--test-days", str(a.test_days),
        "--val-days", str(a.val_days),
        "--min-train-samples", str(a.min_train_samples),
        "--min-eval-samples", str(a.min_eval_samples),
        "--as-of", fold["as_of"],
        "--split-layout", layout,
        "--train-end", train_end,
    ]
    if val_end is not None:
        command += ["--val-end", val_end]

    train_start = fold.get(train_start_key)
    if train_start is not None:
        command += ["--train-start", train_start]

    # 日历列集合是导出规格的一部分：缺省（all）不透传，保持旧命令兼容。
    if getattr(a, "calendar_features", None) is not None:
        command += ["--calendar-features", a.calendar_features]

    return command


def train_command(a, data_root, out_dir, mode, epochs=None, extra=None):
    """构造 stock.training.runner 命令（共享模型参数一份）。

    模型专属参数由适配器的 cli_params 声明；调度器不再写死 TimeXer，
    也不会把 TimeXer 参数透传给其他模型。
    """
    adapter = get_adapter(getattr(a, "model", "TimeXer"))
    command = [
        sys.executable,
        "-m", "stock.training.runner",
        "--model", adapter.name,
        "--mode", mode,
        "--data-root", str(data_root),
        "--out-dir", str(out_dir),
        "--batch-size", str(a.batch_size),
        "--workers", str(a.loader_workers),
        "--lr", str(a.lr),
        "--weight-decay", str(a.weight_decay),
        "--loss", a.loss,
        "--loss-huber-delta", str(a.loss_huber_delta),
        "--seed", str(a.seed),
        "--device", a.device,
    ]
    for name in adapter.cli_params:
        value = getattr(a, name, None)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif name in _BOOLOPT_PARAMS:
                # BooleanOptionalAction 参数：显式传 --no-* 覆盖默认值。
                command.append("--no-" + name.replace("_", "-"))
        else:
            command += [flag, str(value)]
    if epochs is not None:
        command += ["--epochs", str(epochs)]
    if mode == "train":
        command += ["--patience", str(a.patience)]
    if extra:
        command += list(extra)
    return command


def verify_export_config(data_root, expected, stage_label):
    """折边界登记复核：导出登记必须与调度计划完全一致。"""
    exported = json.loads(
        (Path(data_root) / "export_config.json").read_text(encoding="utf-8")
    )
    for key, value in expected.items():
        if exported.get(key) != value:
            raise ValueError(
                f"{stage_label}: 导出登记 {key}={exported.get(key)} "
                f"与计划 {value} 不一致"
            )
    return exported


def load_fold_predictions(out_dir):
    """读取单折测试预测。"""
    return pd.read_csv(
        Path(out_dir) / "test_predictions.csv", dtype={"symbol": str}
    )


def run_pit_universe(a, db, out_path, as_of, stage_label):
    """调用 stock.selection.universe 按历史时点生成部分 PIT 股票池。"""
    command = [
        sys.executable,
        "-m", "stock.selection.universe",
        "--db", db,
        "--out", str(out_path),
        "--as-of", date_text(as_of),
        "--top-mv", str(a.pit_top_mv),
        "--min-years", str(a.pit_min_years),
        "--baidu-fresh-days", str(a.pit_baidu_fresh_days),
    ]
    run_command(command)
    frame = pd.read_csv(out_path, dtype={"symbol": str})
    if frame.empty:
        raise ValueError(
            f"{stage_label}: --as-of {date_text(as_of)} 生成的股票池为空"
        )
    return {
        "path": str(out_path),
        "as_of": date_text(as_of),
        "rows": len(frame),
        "sha256": ds.sha256_file(out_path),
    }


def check_fold_frame(frame, fold, fold_id, stage_label):
    """测试预测日期边界与唯一性校验。"""
    target_dates = pd.to_datetime(frame["target_date"])
    inside = (
        target_dates.ge(pd.Timestamp(fold["test_start"]))
        & target_dates.le(pd.Timestamp(fold["as_of"]))
    )
    if frame.empty or not inside.all():
        raise ValueError(f"{fold_id}/{stage_label}: 测试预测为空或日期越界")

    if frame.duplicated(["symbol", "target_date"]).any():
        raise ValueError(f"{fold_id}/{stage_label}: 单折预测存在重复")


def _finite_or_none(value):
    """NaN/Inf → None（write_json 使用 allow_nan=False）。"""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def aggregate_fold_selection(per_fold):
    """跨折汇总：各折"按天平均"胜率/基线/oracle 的 mean 与 worst。

    worst = 跨折最小值（最差折），用于稳健性报告；不挑选最好折。
    """
    keys = (
        "avg_daily_hit_rate",
        "avg_daily_base_up_rate",
        "avg_oracle_hit_rate",
    )
    out = {}
    for key in keys:
        values = [
            row[key] for row in per_fold
            if isinstance(row.get(key), float) and np.isfinite(row[key])
        ]
        out[key] = {
            "mean": float(np.mean(values)) if values else None,
            "worst": float(np.min(values)) if values else None,
        }
    return out


def selection_summary_block(frame, q_pct, floor_pct):
    """OOS 预测的选股胜率统计（复用 daily_hit，与每日管线同口径）。

    返回 (daily_all, block)：daily_all 为逐日表（含 fold_id）；block 含
    整体/逐折统计与跨折 mean/worst。frame 必须带 fold_id 列。
    """
    daily_parts = []
    per_fold = []
    for fold_id, part in frame.groupby("fold_id"):
        fold_daily, fold_stats = daily_hit(part, q_pct, floor_pct)
        daily_parts.append(fold_daily.assign(fold_id=fold_id))
        per_fold.append({
            "fold_id": fold_id,
            "days": fold_stats["days"],
            "total_signals": fold_stats["total_signals"],
            "pooled_hit_rate": fold_stats["pooled_hit_rate"],
            "avg_daily_hit_rate": fold_stats["avg_daily_hit_rate"],
            "avg_daily_base_up_rate": fold_stats["avg_daily_base_up_rate"],
            "avg_oracle_hit_rate": fold_stats["avg_oracle_hit_rate"],
        })
    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all = daily_all.sort_values("target_date").reset_index(drop=True)
    _, overall = daily_hit(frame, q_pct, floor_pct)
    block = {
        "quantile_pct": q_pct,
        "floor_pct": floor_pct,
        "note": (
            "选股口径与每日管线一致（hybrid_signal：前 q% 且预测 > "
            "floor）；基线为同日同合格集合上涨率；oracle 为按真实收益"
            "完美排序的理论上限参考，不是可实现策略"
        ),
        "overall": {k: _finite_or_none(v) for k, v in overall.items()},
        "per_fold": [
            {k: _finite_or_none(v) for k, v in row.items()}
            for row in per_fold
        ],
        "cross_fold": aggregate_fold_selection(per_fold),
    }
    return daily_all, block


def write_selection_summary(frame, q_pct, floor_pct, out_dir):
    """逐日表落 CSV、统计落 JSON，并返回可并入 oos_summary 的 block。"""
    daily, block = selection_summary_block(frame, q_pct, floor_pct)
    daily.to_csv(Path(out_dir) / "selection_daily.csv", index=False)
    ds.json_write(Path(out_dir) / "selection_summary.json", block)
    return block


def protocol_comparison(control, treatment):
    """协议 A（选择阶段直出）与协议 B（Train+Val 重拟合）的诊断对照。

    只在公共 (symbol, target_date) 上对比模型误差与日度 RankIC，
    并报告覆盖差异；公共样本对照仅用于诊断协议差异，不能替代
    完整流程结果（两协议各自完整覆盖结果才是主要指标）。
    """
    keys = ["symbol", "target_date"]
    left = control[keys + ["y_true", "y_pred"]].rename(
        columns={"y_true": "y_true_selection", "y_pred": "y_pred_selection"}
    )
    right = treatment[keys + ["y_true", "y_pred"]].rename(
        columns={"y_true": "y_true_refit", "y_pred": "y_pred_refit"}
    )
    merged = left.merge(right, on=keys, validate="one_to_one")

    if not merged.empty:
        drift = (
            merged["y_true_selection"] - merged["y_true_refit"]
        ).abs().max()
        if drift > 1e-8:
            raise ValueError(
                "两协议同一 (symbol, target_date) 的 y_true 不一致"
                "（最大差异 %.3e）：数据库可能在运行期间被修订" % drift
            )

    def pooled(pred_col):
        if merged.empty:
            return None
        usable = merged[[pred_col, "y_true_refit"]].dropna()
        if usable.empty:
            return None
        err = usable[pred_col] - usable["y_true_refit"]
        return {
            "mse": float((err ** 2).mean()),
            "mae": float(err.abs().mean()),
        }

    def daily_rank_ic(pred_col):
        # 有限性/退化日过滤：y_true 恒定或预测恒定的交易日 RankIC 无
        # 定义（NaN），必须跳过而不是写进结果（json 写入禁止 NaN，
        # 崩溃会发生在整趟 walk-forward 的最后一步）。
        if merged.empty:
            return None
        grouped = merged.groupby("target_date")
        per_day = []
        for _, day in grouped:
            day = day[[pred_col, "y_true_refit"]].dropna()
            if (
                len(day) < 3
                or day[pred_col].nunique() < 2
                or day["y_true_refit"].nunique() < 2
            ):
                continue
            value = float(
                day[pred_col].rank().corr(day["y_true_refit"].rank())
            )
            if np.isfinite(value):
                per_day.append(value)
        return float(pd.Series(per_day).mean()) if per_day else None

    def coverage(frame):
        return {
            "rows": int(len(frame)),
            "stocks": int(frame["symbol"].nunique()) if len(frame) else 0,
            "dates": (
                int(pd.to_datetime(frame["target_date"]).nunique())
                if len(frame) else 0
            ),
        }

    return {
        "note": (
            "公共样本对照仅用于诊断；主要结论以 oos/test_metrics.json"
            "（协议 B）与 oos/selection_control（协议 A）的完整覆盖"
            "结果为准"
        ),
        "coverage": {
            "selection": coverage(control),
            "refit": coverage(treatment),
            "common": coverage(merged),
        },
        "common_subset": {
            "selection_y_pred": {
                "pooled": pooled("y_pred_selection"),
                "daily_rank_ic": daily_rank_ic("y_pred_selection"),
            },
            "refit_y_pred": {
                "pooled": pooled("y_pred_refit"),
                "daily_rank_ic": daily_rank_ic("y_pred_refit"),
            },
        },
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--db", default=ds.DEFAULT_DB)
    p.add_argument(
        "--universe-csv",
        help="固定输入股票池 CSV；与 --pit-universe 互斥",
    )
    p.add_argument(
        "--pit-universe", action="store_true",
        help="每折调用 python -m stock.selection.universe --as-of 独立生成"
             "部分 PIT 股票池"
             "（选择阶段按 train_end 时点、重拟合阶段按 val_end 时点；"
             "市值排名与简称映射仍是当前快照，见 stock/selection/universe.py）",
    )
    p.add_argument("--pit-top-mv", type=int, default=3000,
                   help="--pit-universe 时的市值排名上限")
    p.add_argument("--pit-min-years", type=float, default=3.0,
                   help="--pit-universe 时的最少历史年限")
    p.add_argument("--pit-baidu-fresh-days", type=int, default=30,
                   help="--pit-universe 时的百度数据新鲜度要求（天）")
    p.add_argument("--run-root", required=True)

    p.add_argument("--first-test-date", required=True)
    p.add_argument("--last-test-date", required=True)

    p.add_argument("--val-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument(
        "--train-days", type=int,
        help="rolling 训练目标交易日数；不传为 expanding",
    )

    p.add_argument(
        "--refit", action="store_true",
        help="启用两阶段协议：Train+Val 重拟合（推荐与缺省协议对照）；"
             "验证集在测试开始前成为历史，可进入最终训练",
    )
    p.add_argument(
        "--refit-window-policy",
        choices=["extend_to_val_end", "keep_train_days"],
        default="extend_to_val_end",
        help="重拟合训练窗口策略：extend_to_val_end=训练起点不变、终点"
             "延伸至 val_end（真正的 Train+Val 合并，expanding 时仍为"
             "全部历史）；keep_train_days=终点为 val_end、起点后移，保持"
             "最近 --train-days 个训练日",
    )

    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument(
        "--model", default="TimeXer",
        help="模型适配器名（stock.adapters 注册）；专属参数由适配器声明",
    )
    p.add_argument("--min-train-samples", type=int, default=250)
    p.add_argument(
        "--min-eval-samples", type=int, default=20,
        help="单股验证/测试覆盖下限登记值（仅写入 manifest.quality."
             "eval_sample_minimum 作报告对照，不决定训练资格；与 "
             "val-days/test-days 的一致性在启动时检查）",
    )
    p.add_argument(
        "--quantile-pct", type=float, default=5.0,
        help="选股胜率统计的逐日排名分位（百分数）；与 "
             "stock.evaluation.daily_hit 缺省口径一致",
    )
    p.add_argument(
        "--floor-pct", type=float, default=0.5,
        help="选股胜率统计的预测涨幅下限（百分点）；与 "
             "stock.evaluation.daily_hit 缺省口径一致",
    )
    p.add_argument("--export-workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=30,
                   help="选择阶段早停上限；重拟合阶段固定预算=各折 best_epoch")
    p.add_argument("--patience", type=int, default=5,
                   help="选择阶段早停耐心（重拟合阶段无早停）")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--loader-workers", type=int, default=0)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--patch-len", type=int, default=10)
    # TimesNet 专属（由适配器 cli_params 声明并透传；其他模型忽略）：
    p.add_argument("--d-ff", type=int,
                   help="TimesNet：TimesBlock Inception 中间宽度，"
                        "缺省 d-model*2")
    p.add_argument("--top-k", type=int, default=5,
                   help="TimesNet：FFT 主周期个数（≤(seq_len+1)//2）")
    p.add_argument("--num-kernels", type=int, default=6,
                   help="TimesNet：Inception 卷积核尺寸数")
    p.add_argument(
        "--use-norm", action="store_true",
        help="TimesNet：开启窗口内 RevIN 归一化（默认关闭；透传给 "
             "stock.training.runner --use-norm，仅作消融对照）",
    )
    p.add_argument(
        "--ex-mask", action="store_true",
        help="屏蔽窗口内整段无观测的外生值 token（透传给 "
             "stock.training.runner --ex-mask，选择/重拟合两阶段一致；"
             "默认关闭＝维持既有\"填补＋缺失标记\"协议，建议以消融"
             "实验对比后再采用）",
    )
    p.add_argument(
        "--calendar-features", default=None,
        help="启用的目标日日历列（3 个时间标记列 + 5 个旗标列；"
             "all/none/逗号列表）；缺省 all，透传给 stock.data.export "
             "--calendar-features。不同集合构成不同导出规格，各折 "
             "data-root 相互独立",
    )
    p.add_argument(
        "--calendar-mode", default=None,
        help="TimeXer 日历列喂入方式（time_marks/columns/none）；"
             "缺省 time_marks（官方通道），透传给 stock.training.runner "
             "--calendar-mode（TimesNet 仅支持 columns）",
    )
    p.add_argument(
        "--head-calendar",
        action=argparse.BooleanOptionalAction, default=None,
        help="TimeXer 预测头拼接目标日星期 one-hot；缺省不透传（= "
             "runner 默认开启），--no-head-calendar 显式关闭并透传",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--loss", choices=["mse", "huber"], default="mse",
        help="训练损失：mse（历史默认）或 huber；透传给 "
             "stock.training.runner，选择/重拟合两阶段一致",
    )
    p.add_argument(
        "--loss-huber-delta", type=float, default=1.0,
        help="huber 损失的 delta（目标为按股标准化收益率，尺度约 1）",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
    )
    return p.parse_args()


def _pit_summary(pit_universe):
    """OOS 结果的 PIT 能力标记（与 plan 的 pit_universe 段同口径）。

    真缺口 #7 的残留补齐：walk_forward_plan.json 已在 --pit-universe
    时写明分类，但拼接后的 oos_summary.json 缺机器可读标记，
    单看摘要文件无法识别评估的 PIT 等级。
    """
    if pit_universe:
        return {
            "universe_mode": "pit_selected",
            "pit_capability": "approximate_pit",
            "pit_criteria": [
                "listed_by_as_of", "quote_history_span_days",
                "quote_days", "has_rzrq", "has_fresh_baidu",
            ],
            "snapshot_criteria": [
                "top_mv_rank", "baidu_keyword_name_mapping",
                "industry_l2", "stock_list_membership",
            ],
            "note": (
                "部分 PIT：上市日期/行情/两融/百度按 as-of 时点评估；"
                "市值排名与简称映射仍是当前快照"
            ),
        }
    return {
        "universe_mode": "fixed_input_pool",
        "pit_capability": "conditional_history",
        "note": (
            "固定输入股票池属于条件性历史评估；"
            "股票池构成不受 as-of 约束（详见 walk_forward_plan.json "
            "的 limitations 与 --pit-universe 说明）"
        ),
    }


def main():
    a = parse_args()

    if min(
        a.val_days, a.test_days, a.seq_len,
        a.min_train_samples, a.min_eval_samples, a.export_workers,
    ) < 1:
        raise ValueError("数量参数必须为正数")

    if not (0.0 < a.quantile_pct <= 100.0) or a.floor_pct < 0:
        raise ValueError("quantile-pct 必须在 (0, 100] 且 floor-pct >= 0")

    # 提前拦截：否则要等昂贵的数据导出跑完、到训练子进程才报错。
    # 校验规则由模型适配器声明（不同模型的内生约束不同）。
    adapter = get_adapter(a.model)
    adapter.validate_args(a, a.seq_len)

    if a.train_days is not None and a.train_days < 1:
        raise ValueError("train-days 必须为正数")

    if min(a.val_days, a.test_days) < a.min_eval_samples:
        raise ValueError(
            "评估块长度小于 min-eval-samples 登记值（配置不一致，"
            "请检查 --val-days/--test-days/--min-eval-samples）"
        )

    if a.refit_window_policy == "keep_train_days" and a.train_days is None:
        raise ValueError("--refit-window-policy=keep_train-days 需要 --train-days")

    if a.pit_universe and a.universe_csv:
        raise ValueError("--pit-universe 与 --universe-csv 互斥")
    if not a.pit_universe and not a.universe_csv:
        raise ValueError("必须提供 --universe-csv 或使用 --pit-universe")
    if a.pit_universe and (
        a.pit_top_mv < 1 or a.pit_min_years <= 0 or a.pit_baidu_fresh_days < 0
    ):
        raise ValueError("pit 股票池参数不合法")

    db = str(Path(a.db).resolve())
    universe = (
        str(Path(a.universe_csv).resolve()) if a.universe_csv else None
    )
    root = Path(a.run_root).resolve()

    if root.exists():
        raise FileExistsError(
            f"{root} 已存在；本调度器不实现断点恢复，请使用新的 run-root"
        )
    if not Path(db).exists():
        raise FileNotFoundError(f"数据库不存在：{db}")
    if universe is not None and not Path(universe).exists():
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
            refit=a.refit,
            refit_window_policy=a.refit_window_policy,
        )

        last_asof = pd.Timestamp(folds[-1]["as_of"])
        if calendar.get_loc(last_asof) + 1 >= len(calendar):
            raise ValueError("日历缺少最后一折 as-of 之后的下一交易日")

        root.mkdir(parents=True)
        (root / "data").mkdir()
        (root / "checkpoints").mkdir()
        (root / "oos").mkdir()

        plan = {
            "model": a.model,
            "protocol": (
                "train_val_refit_v1" if a.refit
                else "independent_train_val_test_v1"
            ),
            "training_window": (
                "rolling" if a.train_days is not None else "expanding"
            ),
            "refit": None if not a.refit else {
                "protocol": "train_val_refit_v1",
                "budget_rule": "same_epochs",
                "budget_rule_note": (
                    "验证阶段确定数据遍历次数，最终在更大的 Train+Val 上"
                    "使用相同遍历次数；相同 epoch 意味着更多优化器更新"
                    "步数；不能用后面折的结果调整前面折的训练预算"
                ),
                "refit_window_policy": a.refit_window_policy,
                "stages": [
                    "1 selection: train_val_test 导出（train<=A、"
                    "val<=B）、独立预处理、从头训练、Val 早停选 "
                    "best_epoch",
                    "2 refit: train_test 导出（fit<=B）、Train+Val 重新"
                    "拟合预处理、从头初始化、固定 best_epoch、无早停",
                    "3 oos: 重拟合模型预测测试块；选择阶段测试预测保留"
                    "为协议 A 对照（oos/selection_control/）",
                ],
                "warm_start": {
                    "implemented": False,
                    "note": (
                        "跨折 warm-start 不在本协议内；每股 scaler/schema"
                        "/特征保留列每折重拟合，上一折权重不能当作无条件"
                        "兼容的连续训练状态；上一折最终模型可能已见过本"
                        "折验证窗口标签，不能用于本折早停选择"
                    ),
                },
            },
            "pit_universe": None if not a.pit_universe else {
                "top_mv": a.pit_top_mv,
                "min_years": a.pit_min_years,
                "baidu_fresh_days": a.pit_baidu_fresh_days,
                "decision_times": (
                    {"selection": "train_end(A)", "refit": "val_end(B)"}
                    if a.refit else {"export": "train_end(A)"}
                ),
                "pit_criteria": [
                    "listed_by_as_of", "quote_history_span_days",
                    "quote_days", "has_rzrq", "has_fresh_baidu",
                ],
                "snapshot_criteria": [
                    "top_mv_rank", "baidu_keyword_name_mapping",
                    "industry_l2", "stock_list_membership",
                ],
                "note": (
                    "部分 PIT：上市日期/行情/两融/百度按 as-of 时点评估；"
                    "市值排名与简称映射仍是当前快照"
                ),
            },
            "args": vars(a),
            "db": db,
            "db_state": baseline_state,
            "universe_csv": universe,
            "universe_sha256": (
                ds.sha256_file(universe) if universe else None
            ),
            "dataset_script_sha256": ds.sha256_file(ds.__file__),
            "select_universe_script_sha256": (
                ds.sha256_file(Path(universe_module.__file__))
                if a.pit_universe else None
            ),
            "export_script_sha256": ds.sha256_file(
                Path(export_module.__file__)
            ),
            "train_script_sha256": ds.sha256_file(
                Path(runner_module.__file__)
            ),
            "folds": folds,
            "actual_last_test_date": folds[-1]["as_of"],
            "limitations": [
                "固定输入股票池属于条件性历史评估；--pit-universe 仅为部分"
                " PIT（市值排名与简称映射仍是当前快照）",
                "无重拟合协议参数截至 train_end；重拟合协议用 Train+Val "
                "缓解，same_epochs 预算下相同 epoch 意味着更多更新步数",
                "每折实际入模股票可能不同（训练期辅助数据覆盖筛选），"
                "按折报告，不做事后交集重训",
                "统计口径收益率评估，不是可实现交易收益",
                "fit_timing 的时限检查只是离线回测的墙上时钟记录，不能"
                "证明当时能按时完成",
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
        control_predictions = []
        fold_metrics = {}
        fit_timing = []

        for fold in folds:
            fold_id = fold["fold_id"]

            # 每折股票池：固定输入或按 A 时点生成的部分 PIT 池。
            if a.pit_universe:
                fold_data_root = root / "data" / fold_id
                selection_universe = fold_data_root / "universe_selection.csv"
                info = run_pit_universe(
                    a, db, selection_universe, fold["train_end"],
                    f"{fold_id}/selection",
                )
                universes = {"selection": info}
            else:
                selection_universe = Path(universe)
                universes = None

            if not a.refit:
                # ===== 无重拟合协议 A（原协议，保留作为对照）=====
                ensure_db_state(
                    con, baseline_state, f"{fold_id} export"
                )

                data_root = root / "data" / fold_id
                out_dir = root / "checkpoints" / fold_id

                run_command(export_command(
                    a, db, selection_universe, data_root, fold,
                    layout="train_val_test",
                    train_end=fold["train_end"], val_end=fold["val_end"],
                ))
                verify_export_config(data_root, {
                    "as_of": fold["as_of"],
                    "train_end": fold["train_end"],
                    "val_end": fold["val_end"],
                    "train_start": fold["train_start"],
                    "split_layout": "train_val_test",
                }, fold_id)

                run_command(train_command(
                    a, data_root, out_dir, mode="train", epochs=a.epochs
                ))

                frame = pd.read_csv(
                    out_dir / "test_predictions.csv", dtype={"symbol": str}
                )
                check_fold_frame(frame, fold, fold_id, "test")

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

                if a.pit_universe:
                    ds.json_write(
                        root / "data" / fold_id / "pit_universes.json",
                        universes,
                    )
                continue

            # ===== 两阶段协议 B：阶段 1 选择 =====
            ensure_db_state(
                con, baseline_state, f"{fold_id} selection export"
            )

            selection_data_root = root / "data" / fold_id / "selection"
            selection_out_dir = root / "checkpoints" / fold_id / "selection"

            run_command(export_command(
                a, db, selection_universe, selection_data_root, fold,
                layout="train_val_test",
                train_end=fold["train_end"], val_end=fold["val_end"],
            ))
            verify_export_config(selection_data_root, {
                "as_of": fold["as_of"],
                "train_end": fold["train_end"],
                "val_end": fold["val_end"],
                "train_start": fold["train_start"],
                "split_layout": "train_val_test",
            }, f"{fold_id}/selection")

            run_command(train_command(
                a, selection_data_root, selection_out_dir,
                mode="train", epochs=a.epochs,
            ))

            # best_epoch 即重拟合阶段的固定训练预算（same_epochs）。
            selection_result = json.loads(
                (selection_out_dir / "selection_result.json").read_text(
                    encoding="utf-8"
                )
            )
            best_epoch = int(selection_result["best_epoch"])
            if best_epoch < 1:
                raise ValueError(f"{fold_id}: 选择阶段 best_epoch 异常")

            # 选择阶段的测试预测 = 协议 A 对照（相同样本外日期）。
            control = load_fold_predictions(selection_out_dir)
            check_fold_frame(control, fold, fold_id, "selection")

            control["fold_id"] = fold_id
            control["train_start"] = fold["train_start"]
            control["train_end"] = fold["train_end"]
            control["val_end"] = fold["val_end"]
            control_predictions.append(control)

            fit_timing.append({
                "fold_id": fold_id,
                "stage": "selection",
                "best_epoch": best_epoch,
                "budget_rule": selection_result.get("budget_rule"),
                "data_cutoff_time": selection_result.get(
                    "fit_data_cutoff_time"
                ),
                "fit_started_at": selection_result.get("fit_started_at"),
                "fit_finished_at": selection_result.get("fit_finished_at"),
                "first_prediction_time": selection_result.get(
                    "first_prediction_time"
                ),
                "deadline_met": selection_result.get(
                    "would_meet_first_prediction_deadline"
                ),
            })

            # ===== 阶段 2 重拟合 =====
            ensure_db_state(
                con, baseline_state, f"{fold_id} refit export"
            )

            if a.pit_universe:
                refit_universe = root / "data" / fold_id / "universe_refit.csv"
                universes["refit"] = run_pit_universe(
                    a, db, refit_universe, fold["val_end"],
                    f"{fold_id}/refit",
                )
            else:
                refit_universe = Path(universe)

            refit_data_root = root / "data" / fold_id / "refit"
            refit_out_dir = root / "checkpoints" / fold_id / "refit"

            run_command(export_command(
                a, db, refit_universe, refit_data_root, fold,
                layout="train_test",
                train_end=fold["val_end"],
                train_start_key="refit_train_start",
            ))
            verify_export_config(refit_data_root, {
                "as_of": fold["as_of"],
                "train_end": fold["val_end"],
                "val_end": None,
                "train_start": fold["refit_train_start"],
                "split_layout": "train_test",
            }, f"{fold_id}/refit")

            refit_extra = [
                "--selection-result", str(
                    selection_out_dir / "selection_result.json"
                ),
                "--refit-window-policy", a.refit_window_policy,
            ]
            if a.pit_universe:
                # 股票池资格决定时点：重拟合池按 B 时点重新确定。
                refit_extra += [
                    "--universe-decision-time", fold["val_end"],
                ]
            run_command(train_command(
                a, refit_data_root, refit_out_dir,
                mode="refit", epochs=best_epoch, extra=refit_extra,
            ))

            treatment = load_fold_predictions(refit_out_dir)
            check_fold_frame(treatment, fold, fold_id, "refit")

            treatment["fold_id"] = fold_id
            treatment["refit_train_start"] = fold["refit_train_start"]
            treatment["fit_end"] = fold["val_end"]
            predictions.append(treatment)

            fit_metadata = json.loads(
                (refit_out_dir / "fit_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            fit_timing.append({
                "fold_id": fold_id,
                "stage": "refit",
                "best_epoch": best_epoch,
                "budget_rule": fit_metadata["training_budget"]["rule"],
                "data_cutoff_time": fit_metadata.get("fit_data_cutoff_time"),
                "fit_started_at": fit_metadata.get("fit_started_at"),
                "fit_finished_at": fit_metadata.get("fit_finished_at"),
                "first_prediction_time": fit_metadata.get(
                    "first_prediction_time"
                ),
                "deadline_met": fit_metadata.get(
                    "would_meet_first_prediction_deadline"
                ),
            })

            fold_metrics[fold_id] = {
                "selection": {
                    "best_epoch": best_epoch,
                    "test_metrics": json.loads(
                        (selection_out_dir / "test_metrics.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                },
                "refit": {
                    "test_metrics": json.loads(
                        (refit_out_dir / "test_metrics.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                },
            }
            print(
                f"\n[FOLD OK] {fold_id}: best_epoch={best_epoch}, "
                f"refit rows={len(treatment)}, "
                f"stocks={treatment['symbol'].nunique()}, "
                f"test={fold['test_start']}~{fold['as_of']}"
            )

            if a.pit_universe:
                ds.json_write(
                    root / "data" / fold_id / "pit_universes.json",
                    universes,
                )

        oos = pd.concat(predictions, ignore_index=True)
        oos = oos.sort_values(
            ["target_date", "symbol"]
        ).reset_index(drop=True)

        if oos.duplicated(["symbol", "target_date"]).any():
            raise ValueError("不同 fold 的 OOS 预测存在重复")

        # 复用训练脚本的报告函数：各折预测已按该折 scaler 还原到原始
        # 收益率尺度，拼接后统一评估；不直接平均各折标准化指标。
        summary = save_test_report(oos, root / "oos")

        ds.json_write(root / "fold_metrics.json", fold_metrics)

        if not a.refit:
            summary["selection"] = write_selection_summary(
                oos, a.quantile_pct, a.floor_pct, root / "oos"
            )
            summary["_pit"] = _pit_summary(bool(a.pit_universe))
            ds.json_write(root / "oos_summary.json", summary)
            print(
                f"\n[OK] folds={len(folds)}, oos_rows={len(oos)}, "
                f"stocks={oos['symbol'].nunique()}, "
                f"dates={oos['target_date'].nunique()}"
            )
            print(f"输出目录：{root}")
            return

        # 协议 A 对照合集（与协议 B 完全相同的 OOS 日期）。
        control_oos = pd.concat(control_predictions, ignore_index=True)
        control_oos = control_oos.sort_values(
            ["target_date", "symbol"]
        ).reset_index(drop=True)
        if control_oos.duplicated(["symbol", "target_date"]).any():
            raise ValueError("不同 fold 的协议 A 对照预测存在重复")

        control_summary = save_test_report(
            control_oos, root / "oos" / "selection_control"
        )
        summary["selection"] = write_selection_summary(
            oos, a.quantile_pct, a.floor_pct, root / "oos"
        )
        control_summary["selection"] = write_selection_summary(
            control_oos, a.quantile_pct, a.floor_pct,
            root / "oos" / "selection_control",
        )
        comparison = protocol_comparison(control_oos, oos)

        ds.json_write(root / "oos" / "comparison.json", comparison)
        ds.json_write(root / "oos_summary.json", {
            "protocol": "train_val_refit_v1",
            "budget_rule": "same_epochs",
            "refit_window_policy": a.refit_window_policy,
            "refit": summary,
            "pit": _pit_summary(bool(a.pit_universe)),
            "selection_control": control_summary,
            "comparison": comparison,
        })
        pd.DataFrame(fit_timing).to_csv(
            root / "fit_timing.csv", index=False
        )

        print(
            f"\n[OK] folds={len(folds)}, "
            f"refit oos_rows={len(oos)}, "
            f"control rows={len(control_oos)}, "
            f"stocks={oos['symbol'].nunique()}, "
            f"dates={oos['target_date'].nunique()}"
        )
        print(f"输出目录：{root}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
