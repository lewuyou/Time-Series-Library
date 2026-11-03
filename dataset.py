# -*- coding: utf-8 -*-
"""
DuckDB -> TimeXer 单股票下一交易日收益率数据。

数据库默认位于 ../AutoTS/download/autots.duckdb；交易日历优先从库内
holiday_calendar 推导（is_holiday_related 即休市标记），无需外部 CSV。

支持：
    1. 单股 / 行业 / 市值排序选股
    2. 行情、百度、估值、两融、宏观 close + volume
    3. 可用时间假设对齐，禁止 bfill
    4. 训练区间拟合预处理
    5. 连续矩阵 + 样本索引，避免展开全部历史窗口
    6. 独立 infer 模式，冻结特征和预处理参数
    7. 附带 PyTorch Dataset，训练时直接读取导出结果
    8. 两种样本划分布局（--split-layout，见 SPLIT_LAYOUTS）：
       train_val_test（默认，原协议：train/val/test 三分）；
       train_test（walk-forward 两阶段协议的重拟合导出：Train+Val 合并
       为训练集、无验证集，--train-end 即最终训练截止日 fit_end）。
       两种布局下单股只强制训练样本下限；验证/测试覆盖率只登记在
       manifest.quality，不决定训练资格（未来评估期发生停牌/缺数/退市
       不应反过来剔除该股票的训练样本，避免测试期筛选偏差）。
       samples.csv 自 v7 起区分两类口径：输入窗口有效（可预测）的行全部
       保留，label_valid=False 表示目标日无有效标签（可预测、不可评分，
       y_raw 为空），监督训练/评估装载时只取 label_valid=True 的行。

预测时点：
    历史行情截至交易日 t。
    在下一交易日 t+1 的北京时间 08:30 形成预测。
    预测 t+1 相对 t 的后复权收盘收益率。

每个历史特征行的辅助数据均按该行自己的预测截止时间对齐。
不会在后来的预测时点追溯更新过去各行的辅助数据快照。

交易日历：
    --calendar 提供 CSV 时以 CSV 为准（date 列，需覆盖数据区间并包含
    as-of 之后的下一交易日）。否则从数据库推导：
      a) holiday_calendar 中非 is_holiday_related 的周一至周五为候选；
      b) 历史区间再用 akshare_tx 当日有行情的股票数（>=
         --min-market-symbols）校准，剔除日历表漏标的休市日
         （如 2020-01-31、2024-02-09）；
      c) 最新行情之后的未来交易日只能采用 a) 的推导结果。
    默认 --calendar-start 2016-01-01，即只使用 2016 年以后的数据，
    更早的历史不加载。

训练导出（每只股票一个子目录）：
    aligned_values.csv       对齐后的原值，未进行 log/标准化
    features_unscaled.csv    派生与固定变换后的特征
    alignment_audit.csv      辅助数据来源时间审计
    feature_matrix.npy       [T, F]，float32，最终标准化矩阵
    dates.npy                [T]，交易日期
    samples.csv              历史窗口位置、划分、label_valid、标签
                             （label_valid=False 的行为"可预测但目标日
                             无有效标签"，y_raw/y_scaled 为空，仅登记）
    train_preview.csv        训练前数据预览：逐交易日的标准化矩阵
                             （与 feature_matrix.npy 逐值一致）+ 当日原始
                             收益率 y_raw + 按目标日期口径的 train/val/test
                             划分，供训练前人工抽查数据形态
    latest.npz               [1, L, F]，最新有效推理窗口
    preprocessing.json       列顺序、填补和标准化参数、日历指纹
    manifest.json            配置、质量统计、来源限制、水位与警告

导出后校验（不连数据库，纯文件检查）：
    python dataset.py --mode validate --data-dir dataset/timexer_run_001/601398
    覆盖：样本索引/标签一致性、y_raw 由收盘价重算、y_scaled 与矩阵末列
    一致、train_preview.csv 与矩阵逐值一致、切分边界、审计
    available<=decision、验证/测试数据不影响预处理（扰动后重拟合统计量
    不变）、未来行扰动不影响历史矩阵。

infer 模式的历史快照一致性：
    infer 只冻结预处理参数，历史特征行仍按当前数据库重算。若数据库中的
    历史值发生修订（如 TradingView 全历史重抓覆盖、财务回算），重算出的
    历史行会与训练期不同。infer 会与训练期 feature_matrix.npy 按日期逐行
    比对并把漂移写入 manifest.history_drift 与 warnings；加
    --fail-on-history-drift 可把漂移升级为硬错误。

字段单位（依据 ../AutoTS/download/sources 各采集模块的入库说明）：
    rzrq 比率字段原单位为百分数，/100.0 转为小数是正确的；
    akshare_tx.volume 单位为股，特征使用 log1p；turnover 本身即小数。

目标日星期 one-hot（target_dow_mon..fri，v8 起）：
    每行的目标日 = 下一交易日（该行作为输入日时的预测目标）。
    星期是已知的日历信息，不引入未来数据；样本窗口最后一行
    （决策行）即该样本目标日的星期，与 latest.npz 的 target_date
    口径一致。one-hot 保持 0/1，不标准化、恒不缺失。

运行示例：
    # 单股训练数据（as-of 默认取库内最新行情日）
    python dataset.py --symbol 601398 \\
        --train-end 2023-12-31 --val-end 2024-12-31 \\
        --out-dir dataset/timexer_run_001

    # 行业内市值前 10
    python dataset.py --industry 电力 --top 10 \\
        --train-end 2023-12-31 --val-end 2024-12-31 \\
        --continue-on-error --out-dir dataset/timexer_power_top10

    # 指定宏观品种
    python dataset.py --symbol 601398 \\
        --macro-symbols SSE:000001 SSE:510300 SPCFD:SPX \\
        --train-end 2023-12-31 --val-end 2024-12-31 \\
        --out-dir dataset/timexer_macro_001

    # 复用已有预处理参数生成最新预测窗口
    python dataset.py --mode infer \\
        --preprocessor dataset/timexer_run_001/601398/preprocessing.json \\
        --out-dir dataset/timexer_infer_001

训练读取：
    from dataset import StockTimeXerDataset
    ds = StockTimeXerDataset("dataset/timexer_run_001/601398", split="train")
    # batch_x / batch_y 形状分别为 [B, seq_len, F] / [B, 1, 1]
    # 模型调用：prediction = model(batch_x, None, None, None)
    # 还原：raw = prediction * scale + mean（见 preprocessing.json）
    #       若 return_type=log，raw 仍是对数收益率，简单收益率用 expm1(raw)，
    #       也可直接调用 ds.to_simple_return(prediction)。

训练脚本：
    共享训练使用 train_global_timexer.py（独立窗口、联合训练），
    不依赖本模块，只读导出文件。注意 TimeXer 内生分支用不重叠 patch：
    patch_len 必须整除 seq_len，否则 60/16 会丢弃最近 12 天目标历史，
    训练脚本已内置该校验。
    本模块底部的 StockTimeXerDataset 保留用于单股训练的独立训练循环
    （features="MS"、pred_len=1、label_len=0，只读导出文件）。

共享模型流程中的定位（完整流程见 export_global.py 头部说明）：
    1. select_universe.py 粗筛股票池；
    2. export_global.py 批量导出（复用本模块 prepare_one）并复检
       baidu_/rzrq_ 特征，产出 export_results.csv；
    3. train_global_timexer.py 联合训练（不依赖本模块，只读导出文件）；
    4. 本模块 --mode validate 抽查导出质量；--mode infer 冻结训练期
       预处理参数生成最新推理窗口，供 train_global_timexer.py
       --mode predict 使用（同一股票必须复用训练期 scaler）。

注意：
    这是自定义 Dataset 数据格式，不是 Dataset_Custom 的直接输入。
    不要对 feature_matrix.npy 再做标准化，也不要把 y_raw 或 y_scaled
    加到输入特征里。
    可用时间假设对齐（观测日期 + 假设发布滞后）比直接按日期连接合理，
    但原表没有真实历史发布时间，不能严格证明历史当时可见；百度指数
    实际延迟、两融发布时刻、跨时区行情、宏观修订、历史复权更新需人工核实。
    当前股票列表、行业、市值用于选择处理对象，不作为历史数值特征。
    预测的是统计口径的下一交易日收盘收益率；t+1 日 08:30 产生信号后
    无法再按 t 日收盘价成交，该指标不能直接当作可实现交易收益。
"""

# ------------------------------------------------------------------
# 以下实现已按职责拆分到 stock/data/*（函数体逐字迁出，本文件退化为
# 薄 CLI 入口 + 兼容转发）。所有旧函数名在此仍可 import，供
# export.py 的 `import dataset as ds`、tests 与外部脚本继续使用。
#
# dataset_script_sha256 兼容规则（walk_forward/runner 登记，export 写入
# export_config.json）：
#   1) 该哈希 = sha256(本文件)；本文件任何改动（含注释）都会变化，
#      属预期 —— 它只是导出脚本身份指纹，不是数据契约；
#   2) 旧 data-root 的 --mode validate / --mode infer 不比对该哈希，
#      拆分后照常可用（契约由 feature_schema_hash / 日历指纹等承担）；
#   3) 只有对已存在导出目录原地重导会被 prepare_one 的
#      FileExistsError 拒绝：口径变更必须换新 --out-dir（新 data-root），
#      不要复用旧目录混入新代码的产物；
#   4) 判断两份导出是否同口径，请比对 manifest 的 feature_schema、
#      日历指纹与 config，而不是 dataset_script_sha256。
# ------------------------------------------------------------------

import argparse
import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# torch 仅 StockTimeXerDataset（单股独立训练循环）需要，预处理可不安装。
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# 拆分后的实现模块（单一事实源）；名字经此转发保持旧 API 兼容。
from stock.data.calendar import (
    calendar_dates_sha256,
    calendar_fingerprint,
    load_calendar,
    read_calendar,
)
from stock.data.common import (
    BAIDU_FIELDS,
    COMPATIBLE_VERSIONS,
    DEFAULT_CALENDAR_START,
    DEFAULT_POLICIES,
    DEFAULT_DB,
    PRICE_FIELDS,
    RZRQ_FIELDS,
    SPLIT_LAYOUTS,
    VAL_FIELDS,
    VERSION,
    as_numeric,
    assert_unique,
    json_write,
    load_policies,
    norm_symbol,
    parse_dates,
    sql_symbol,
)
from stock.data.pit import align_one, history_drift_report, source_watermarks
from stock.data.builder import (
    coverage_report,
    load_asof,
    prepare_one,
    save_latest,
)
from stock.data.features import build_features, fixed_transform
from stock.data.preprocessing import fit_preprocessing, transform
from stock.data.schema import (
    TARGET,
    TARGET_DOW_NAMES,
    age_col,
    dow_column,
    feature_schema_doc,
    is_companion_col,
    is_dow_col,
    is_exogenous_col,
    is_missing_col,
    missing_col,
)
from stock.data.sources import (
    empty_series,
    load_macro,
    load_prices,
    load_stock_aux,
    macro_column,
    make_specs_from_long,
    select_stocks,
)
from stock.data.targets import (
    label_valid as is_label_valid,
    target_series,
    target_spec,
)
from stock.data.validation import validate_dataset
from stock.data.windows import make_samples
from stock.utils import sha256_file, write_json

# resolve_split 不做顶层转发：splits.py 顶层 ，
# 若本模块顶层再导入 splits 会形成加载环（ 时
# __main__ 与名字 dataset 是两个模块实例）。用 PEP 562 模块级
# __getattr__ 延迟转发：
# 与 inspect.getsource(ds.resolve_split) 均可用（指向 splits.py）。
def __getattr__(name):
    if name == "resolve_split":
        from stock.data.splits import resolve_split

        return resolve_split
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["prepare", "infer", "validate"], default="prepare",
    )
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument(
        "--data-dir",
        help="validate 必填：已导出的股票数据目录（如 "
             "dataset/timexer_run_001/601398）",
    )
    p.add_argument(
        "--calendar",
        help="可选交易日历 CSV（date 列）；缺省时从数据库推导",
    )
    p.add_argument(
        "--calendar-start", default=DEFAULT_CALENDAR_START,
        help=f"日历起点，早于该日的历史不加载（默认 {DEFAULT_CALENDAR_START}）",
    )
    p.add_argument(
        "--min-market-symbols", type=int, default=50,
        help="数据库日历校准：当日至少多少只股票有行情才算交易日",
    )
    p.add_argument("--symbol")
    p.add_argument("--industry")
    p.add_argument("--top", type=int)
    p.add_argument("--baidu-keyword")
    p.add_argument("--as-of")
    p.add_argument("--start", help="最早样本目标日期，不截断预热行情")
    p.add_argument("--tail", type=int, help="最近 N 个候选目标交易日")

    p.add_argument("--train-end")
    p.add_argument("--val-end")
    p.add_argument(
        "--split-layout", choices=list(SPLIT_LAYOUTS),
        default="train_val_test",
        help="样本划分布局：train_val_test（原协议，需 --val-end）；"
             "train_test（walk-forward 重拟合导出，无验证集，"
             "--train-end 即最终训练截止日 fit_end，不得提供 --val-end）",
    )
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--return-type", choices=["simple", "log"], default="simple")
    p.add_argument(
        "--min-train-samples", type=int, default=250,
        help="单股可评分训练样本（label_valid）下限，低于该值的股票导出"
             "失败（训练资格只由训练期决定）",
    )
    p.add_argument(
        "--min-eval-samples", type=int, default=20,
        help="单股验证/测试覆盖下限（仅登记到 manifest.quality."
             "eval_sample_minimum 作报告对照，不决定训练资格）",
    )
    p.add_argument("--min-feature-coverage", type=float, default=0.60)
    p.add_argument("--min-observations", type=int, default=100)

    p.add_argument(
        "--macro-fields", nargs="+", choices=["close", "volume"],
        default=["close", "volume"],
    )
    p.add_argument(
        "--macro-symbols", nargs="+",
        help="不指定时读取查询区间内全部宏观品种",
    )
    p.add_argument("--source-policy")
    p.add_argument("--decision-hour", type=int, default=8)
    p.add_argument("--decision-minute", type=int, default=30)
    p.add_argument(
        "--preprocessor",
        help="infer 必填：已有 preprocessing.json 路径",
    )
    p.add_argument(
        "--fail-on-history-drift", action="store_true",
        help="infer：历史特征行与训练期不一致时按错误处理（默认仅告警并"
             "写入 manifest.history_drift）",
    )
    p.add_argument(
        "--allow-calendar-drift", action="store_true",
        help="infer：训练期日历前缀与当前不一致时降级为告警（默认报错）",
    )
    p.add_argument("--out-dir", default="dataset/timexer_prepared")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()




def main():
    args = parse_args()

    if args.symbol and args.industry:
        raise ValueError("--symbol 与 --industry 二选一")
    if args.baidu_keyword and not args.symbol:
        raise ValueError("--baidu-keyword 仅支持单股")
    if args.top is not None and args.top < 1:
        raise ValueError("--top 必须 > 0")
    if args.tail is not None and args.tail < 1:
        raise ValueError("--tail 必须 > 0")
    if args.seq_len < 2:
        raise ValueError("--seq-len 必须 >= 2")
    if not 0 <= args.min_feature_coverage <= 1:
        raise ValueError("--min-feature-coverage 必须在 0~1")
    if min(
        args.min_train_samples, args.min_eval_samples, args.min_observations
    ) < 1:
        raise ValueError("最小样本量和观测量必须为正数")
    if not 0 <= args.decision_hour <= 23:
        raise ValueError("decision-hour 非法")
    if not 0 <= args.decision_minute <= 59:
        raise ValueError("decision-minute 非法")
    try:
        pd.Timestamp(args.calendar_start)
    except ValueError as exc:
        raise ValueError(f"--calendar-start 日期非法：{exc}") from exc

    if args.mode == "validate":
        if not args.data_dir:
            raise ValueError("validate 必须提供 --data-dir")
        ok = validate_dataset(args.data_dir)
        raise SystemExit(0 if ok else 1)

    if not Path(args.db).exists():
        raise FileNotFoundError(f"数据库不存在：{args.db}")

    frozen = None
    if args.mode == "infer":
        if not args.preprocessor:
            raise ValueError("infer 必须提供 --preprocessor")
        if args.industry or args.top:
            raise ValueError("infer 每次对应一个已训练股票，请勿指定行业/top")
        frozen = json.loads(
            Path(args.preprocessor).read_text(encoding="utf-8")
        )
        if frozen["version"] not in COMPATIBLE_VERSIONS:
            raise ValueError(
                f"预处理版本 {frozen['version']} 与脚本版本 {VERSION} 不兼容"
            )
        if frozen["version"] != VERSION:
            print(
                f"[WARN] 预处理文件为 {frozen['version']}（当前 {VERSION}），"
                "按兼容逻辑读取；旧文件可能缺少日历指纹、sample_selection "
                "等新增元信息，相关校验会回退"
            )
        if args.symbol and norm_symbol(args.symbol) != frozen["symbol"]:
            raise ValueError("指定股票与预处理文件不一致")
        args.symbol = frozen["symbol"]
        config = frozen["config"]
    else:
        if not args.train_end:
            raise ValueError("prepare 必须提供 train-end")
        if args.split_layout == "train_val_test":
            if not args.val_end:
                raise ValueError(
                    "train_val_test 布局必须提供 train-end 和 val-end"
                )
            if pd.Timestamp(args.train_end) >= pd.Timestamp(args.val_end):
                raise ValueError("必须满足 train-end < val-end")
        elif args.val_end:
            raise ValueError(
                "train_test 布局没有验证集，不能提供 val-end"
            )

        config = {
            "seq_len": args.seq_len,
            "return_type": args.return_type,
            "macro_fields": sorted(set(args.macro_fields)),
            "macro_symbols": args.macro_symbols,
            "policies": load_policies(args.source_policy),
            "decision_hour": args.decision_hour,
            "decision_minute": args.decision_minute,
            "baidu_keyword": args.baidu_keyword,
        }

    results = []
    con = duckdb.connect(args.db, read_only=True)
    try:
        con.execute("BEGIN TRANSACTION")

        frozen_calendar = frozen.get("calendar") if frozen else None
        if frozen_calendar:
            # infer 冻结训练期日历参数，命令行日历参数只在其缺失时生效。
            if frozen_calendar.get("source") == "csv" and frozen_calendar.get("path"):
                cal_path = frozen_calendar["path"]
                if not Path(cal_path).exists():
                    raise FileNotFoundError(
                        f"训练期日历 CSV 不存在：{cal_path}"
                    )
                current_sha = sha256_file(cal_path)
                if current_sha != frozen_calendar.get("sha256_of_calendar_csv"):
                    # 全文件 sha 只作为版本记录；合法变化（在训练 as-of
                    # 之后追加新交易日）由下方历史前缀校验兜底，这里提示
                    # 而不是拒绝，避免阻断滚动追加。
                    print(
                        "[WARN] 日历 CSV 全文件 sha256 与训练期不同"
                        "（可能追加了训练 as-of 之后的新交易日），"
                        "将以历史前缀校验结果为准"
                    )
                cal_start = frozen_calendar.get("calendar_start")
                cal_min = frozen_calendar.get("min_market_symbols")
            else:
                if args.calendar:
                    print(
                        "[WARN] infer 忽略 --calendar：预处理文件冻结了"
                        "数据库推导日历"
                    )
                cal_path = None
                cal_start = frozen_calendar.get("calendar_start")
                cal_min = frozen_calendar.get("min_market_symbols")
                if (
                    args.calendar_start != DEFAULT_CALENDAR_START
                    and args.calendar_start != cal_start
                ):
                    print(
                        f"[WARN] infer 忽略 --calendar-start="
                        f"{args.calendar_start}，使用训练期 {cal_start}"
                    )
                if cal_min is not None and args.min_market_symbols != cal_min:
                    print(
                        f"[WARN] infer 忽略 --min-market-symbols="
                        f"{args.min_market_symbols}，使用训练期 {cal_min}"
                    )
            calendar, calendar_info = load_calendar(
                con, cal_path, cal_start, cal_min
            )
            # 历史前缀必须与训练期一致；只允许追加训练 as-of 之后的新交易日。
            sha, days = calendar_dates_sha256(
                calendar, frozen_calendar["as_of"]
            )
            if (
                sha != frozen_calendar.get("sha256")
                or days != frozen_calendar.get("trading_days_up_to_as_of")
            ):
                msg = (
                    "交易日历历史前缀与训练期不一致"
                    f"（{days} vs {frozen_calendar.get('trading_days_up_to_as_of')}"
                    " 天）：日历表或行情覆盖发生变化，历史窗口口径改变"
                )
                if args.allow_calendar_drift:
                    print(f"[WARN] {msg}（--allow-calendar-drift 已启用）")
                else:
                    raise ValueError(
                        msg + "；确认无害可加 --allow-calendar-drift"
                    )
        else:
            calendar, calendar_info = load_calendar(
                con, args.calendar, args.calendar_start,
                args.min_market_symbols,
            )
        as_of = load_asof(args, con, calendar)
        if (
            args.mode == "prepare"
            and args.split_layout == "train_val_test"
            and pd.Timestamp(args.val_end) >= as_of
        ):
            raise ValueError("必须满足 val-end < as-of，以保留测试区间")
        if (
            args.mode == "prepare"
            and args.split_layout == "train_test"
            and pd.Timestamp(args.train_end) > as_of
        ):
            raise ValueError("必须满足 train-end <= as-of")

        stocks = select_stocks(
            con, args.symbol, args.industry, args.top
        )
        max_lookback = max(
            x["max_age_days"] + x["lag_days"]
            for x in config["policies"].values()
        )
        aux_start = calendar.min() - pd.Timedelta(days=max_lookback + 10)
        next_session = calendar[calendar.get_loc(as_of) + 1]

        macro = load_macro(
            con, config["macro_fields"], config["macro_symbols"],
            aux_start, next_session,
        )

        # 训练时未指定宏观白名单，则冻结本次实际品种列表。
        if frozen is None:
            config["macro_symbols"] = sorted({
                x["symbol"] for x in macro
            })

        for row in stocks.to_dict("records"):
            try:
                results.append(prepare_one(
                    con, row, args, calendar, as_of, macro, config,
                    calendar_info, frozen
                ))
            except Exception as exc:
                print(f"[ERROR] {row['symbol']}：{exc}")
                results.append({
                    "symbol": row["symbol"],
                    "status": "error",
                    "error": str(exc),
                })
                if not args.continue_on_error:
                    raise
        con.execute("COMMIT")
    finally:
        con.close()
        if results:
            root = Path(args.out_dir)
            root.mkdir(parents=True, exist_ok=True)
            name = datetime.now().strftime("run_summary_%Y%m%d_%H%M%S_%f.json")
            json_write(root / name, results)




if HAS_TORCH:

    class StockTimeXerDataset(_TorchDataset):
        """读取 dataset.py 导出的单只股票训练数据。

        x: [seq_len, F] float32，已标准化的输入窗口；
        y: [1, 1] float32，标准化后的下一交易日收益率。

        时间序列库的 Dataset 返回四元组 (x, y, x_mark, y_mark)，本类为
        股票任务定制返回二元组；请配合 train_timexer.py 的独立训练循环
        使用，不要接入 run.py / exp 循环。
        """

        def __init__(self, root, split="train"):
            root = Path(root)
            if split not in {"train", "val", "test"}:
                raise ValueError(f"非法 split：{split}")

            self.root = root
            self.matrix = np.load(
                root / "feature_matrix.npy", mmap_mode="r"
            )
            samples = pd.read_csv(root / "samples.csv")
            if "split" not in samples:
                raise ValueError("samples.csv 缺少 split 列")
            if "label_valid" not in samples:
                raise ValueError(
                    "samples.csv 缺少 label_valid 列（旧版导出），"
                    "请用当前版本重新导出"
                )
            # 只装载可评分样本（label_valid）：无标签行 y_scaled 为空，
            # 不能作为监督样本。
            self.samples = samples[
                (samples["split"] == split) & samples["label_valid"]
            ]
            self.samples = self.samples.reset_index(drop=True)
            if self.samples.empty:
                raise ValueError(f"{root} 没有 {split} 样本")

            with open(root / "preprocessing.json", encoding="utf-8") as f:
                self.preprocessing = json.load(f)
            self.seq_len = int(self.preprocessing["config"]["seq_len"])
            self.return_type = self.preprocessing["config"].get(
                "return_type", "simple"
            )
            self.feature_count = self.matrix.shape[1]

            row = self.samples.iloc[0]
            span = int(row.end_pos) - int(row.start_pos) + 1
            if span != self.seq_len:
                raise ValueError(
                    f"样本窗口长度 {span} 与 seq_len {self.seq_len} 不一致"
                )

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            row = self.samples.iloc[idx]
            x = np.array(
                self.matrix[int(row.start_pos):int(row.end_pos) + 1],
                dtype=np.float32,
            )
            y = np.array([[float(row.y_scaled)]], dtype=np.float32)
            return x, y

        def inverse_target(self, values):
            """把标准化输出还原为标签空间的原始值。

            return_type=simple 时即简单收益率；return_type=log 时仍是
            对数收益率（与 y_raw 同空间），要得到简单收益率请用
            to_simple_return。
            """
            p = self.preprocessing["statistics"][TARGET]
            return values * p["scale"] + p["mean"]

        def to_simple_return(self, values):
            """把标准化输出还原为简单收益率（对数标签自动 expm1）。

            values 可以是 numpy 数组或 torch 张量（评估工具会传张量），
            expm1 相应选择实现。
            """
            raw = self.inverse_target(values)
            if self.return_type == "log":
                if HAS_TORCH and torch.is_tensor(raw):
                    return torch.expm1(raw)
                return np.expm1(raw)
            return raw




if __name__ == "__main__":
    main()
