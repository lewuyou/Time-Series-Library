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

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# 特征列布局（__missing / __age_days / target_dow_* 命名与判定）的
# 单一事实源在 stock/data/schema.py：本文件（生成侧）与训练/推理侧
# （解析侧的张量对齐函数）共用同一实现，改特征方案只改那一处。
from stock.data.schema import (
    TARGET,
    TARGET_DOW_NAMES,
    age_col,
    dow_column,
    is_companion_col,
    is_dow_col,
    is_exogenous_col,
    is_missing_col,
    missing_col,
)
# JSON 原子写 / 文件哈希：全仓唯一实现（旧 json_write/sha256_file
# 本地副本删除，名字保留别名以兼容 ds.json_write 调用方）。
from stock.utils import sha256_file, write_json

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    HAS_TORCH = True
except Exception:  # torch 仅训练读取需要，预处理可不安装
    HAS_TORCH = False


VERSION = "stock_timexer_v8"
# 可读取的历史版本：旧版缺少日历指纹、sample_selection、split_layout、
# label_valid 等新增字段，infer 按兼容逻辑回退旧行为并提示。
# v7 起 samples.csv 区分"可预测"（input_eligible，行保留）与"可评分"
# （label_valid=True），且单股仅强制训练样本下限，验证/测试覆盖率只登记。
# v8 起特征加入目标日星期 one-hot（target_dow_mon..fri，保持 0/1
# 不标准化）；旧版 preprocessor infer 仍兼容（one-hot 列被计算但不使用）。
COMPATIBLE_VERSIONS = (
    "stock_timexer_v3", "stock_timexer_v4", "stock_timexer_v5",
    "stock_timexer_v6", "stock_timexer_v7", "stock_timexer_v8",
)

# 样本划分布局：
#   train_val_test：train <= train_end < val <= val_end < test（原协议）；
#   train_test：   train <= train_end（即 fit_end，含验证区间）< test。
# train_test 用于 walk-forward 两阶段协议的重拟合导出：验证集在测试开始前
# 已成为历史，可以进入最终训练；测试集不参与任何训练期决策。
SPLIT_LAYOUTS = ("train_val_test", "train_test")
# TARGET / TARGET_DOW_NAMES 改自 stock.data.schema 导入（单一事实源），
# 值与历史版本一致。

DEFAULT_DB = "../AutoTS/download/autots.duckdb"
DEFAULT_CALENDAR_START = "2016-01-01"

# volume 单位为股（采集端已从手换算），特征侧用 log1p 压缩量级。
PRICE_FIELDS = ["open", "close", "high", "low", "volume", "turnover"]
BAIDU_FIELDS = ["search_all", "feed"]
VAL_FIELDS = ["pe", "pb", "ps", "pcf"]
RZRQ_FIELDS = ["fin_tval_ratio", "loan_tval_ratio"]

# 可覆盖来源策略，也可精确覆盖到 "macro:品种:字段"。
DEFAULT_POLICIES = {
    "baidu": {"lag_days": 2, "release_hour": 0, "max_age_days": 7},
    "guzhi": {"lag_days": 2, "release_hour": 0, "max_age_days": 10},
    "rzrq": {"lag_days": 2, "release_hour": 0, "max_age_days": 5},
    "macro": {"lag_days": 2, "release_hour": 0, "max_age_days": 10},
}


def norm_symbol(value):
    text = re.sub(r"^(sh|sz)", "", str(value).strip().lower())
    if not re.fullmatch(r"\d{1,6}", text):
        raise ValueError(f"非法股票代码：{value!r}")
    return text.zfill(6)


def sql_symbol(column):
    return (
        "lpad(regexp_replace("
        f"lower(trim(cast({column} as varchar))), "
        "'^(sh|sz)', ''), 6, '0')"
    )


json_write = write_json


def parse_dates(df):
    df = df.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("原始数据存在空日期")
    return df


def assert_unique(df, keys, name):
    if df.empty:
        return
    bad = df.duplicated(keys, keep=False)
    if bad.any():
        example = df.loc[bad, keys].head(8).to_dict("records")
        raise ValueError(f"{name} 存在重复业务键：{example}")


def as_numeric(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def read_calendar(path):
    df = pd.read_csv(path)
    if "date" not in df:
        raise ValueError("交易日历必须包含 date 列")
    df = parse_dates(df)
    assert_unique(df, ["date"], "交易日历")
    idx = pd.DatetimeIndex(df["date"].sort_values(), name="date")
    if len(idx) < 2:
        raise ValueError("交易日历太短")
    return idx


def load_calendar(con, path=None, start=None, min_market_symbols=50):
    """返回 (交易日历, 日历来源说明)。

    CSV 模式以 --calendar 为准；数据库模式见模块 docstring。
    两种模式最终都截断到 start（默认 2016-01-01）之后。

    数据库模式的原则：交易日历回答"某天应不应该交易"，行情覆盖率只反映
    "某天数据完不完整"，两者不混用：
      a) 当日完全无行情的推导交易日才从日历剔除（多为 holiday_calendar
         漏标的休市日，如 2020-01-31、2024-02-09），剔除日逐日记录在
         info.removed_no_quote_days，便于人工区分休市与采集故障；
      b) 当日有部分行情（0 < 覆盖 < --min-market-symbols）视为数据异常的
         真实交易日，保留在日历并记录在 info.partial_coverage_days，
         不再静默删除（否则会把跨多日的收益率当成单日收益率）；
      c) 行情充足但被 holiday_calendar 标记为休市的日期记入
         info.quoted_days_missing_from_derived（目标股票当日有行情时会在
         build_features 报"行情日期不在交易日历"）。
    """
    if path:
        idx = read_calendar(path)
        info = {
            "source": "csv",
            "path": str(path),
            "sha256": sha256_file(path),
        }
    else:
        derived = con.execute(
            """
            SELECT date FROM holiday_calendar
            WHERE NOT is_holiday_related
              AND isodow(date) BETWEEN 1 AND 5
            ORDER BY date
            """
        ).fetchdf()
        if derived.empty:
            raise ValueError("holiday_calendar 无法推导交易日历")
        derived = pd.DatetimeIndex(
            pd.to_datetime(derived["date"])
        ).unique().sort_values()

        # 逐日行情覆盖数：>= 阈值视为齐全交易日，> 0 视为真实交易日。
        counts = con.execute(
            """
            SELECT date, count(DISTINCT symbol) AS n
            FROM akshare_tx
            WHERE adjust = 'hfq'
            GROUP BY date
            ORDER BY date
            """
        ).fetchdf()
        if counts.empty:
            raise ValueError("数据库没有可用于校准的行情日期")
        counts["date"] = pd.to_datetime(counts["date"])
        n_by_date = pd.Series(
            counts["n"].to_numpy(), index=pd.DatetimeIndex(counts["date"])
        )
        last_actual = n_by_date.index.max()

        quoted = n_by_date[n_by_date >= min_market_symbols].index
        partial = n_by_date[(n_by_date > 0) & (n_by_date < min_market_symbols)]
        historical = derived[derived <= last_actual]

        kept = historical[
            historical.isin(quoted) | historical.isin(partial.index)
        ]
        idx = kept.union(derived[derived > last_actual]).sort_values()

        start_ts = (
            pd.Timestamp(start).normalize() if start is not None else None
        )
        removed = [
            {
                "date": str(d.date()),
                "quoted_symbols": int(n_by_date.get(d, 0)),
            }
            for d in historical[
                ~historical.isin(quoted) & ~historical.isin(partial.index)
            ]
            if start_ts is None or d >= start_ts
        ]
        partial_days = [
            {
                "date": str(d.date()),
                "quoted_symbols": int(n),
            }
            for d, n in partial.items()
            if d in derived and (start_ts is None or d >= start_ts)
        ]
        extra = quoted[
            ~quoted.isin(derived)
            & (start_ts is None or (quoted >= start_ts))
        ]

        info = {
            "source": "db",
            "note": (
                "历史交易日经 akshare_tx 行情校准（完全无行情才剔除，"
                "部分覆盖保留并标记）；"
                f"{last_actual.date()} 之后的交易日由 holiday_calendar 推导"
            ),
            "min_market_symbols": int(min_market_symbols),
            "removed_no_quote_days": removed,
            "partial_coverage_days": partial_days,
            "quoted_days_missing_from_derived": {
                "count": int(len(extra)),
                "examples": [str(d.date()) for d in extra[:5]],
            },
        }

    if start is not None:
        start = pd.Timestamp(start).normalize()
        idx = idx[idx >= start]
        info["calendar_start"] = str(start.date())
    if len(idx) < 2:
        raise ValueError("截断后交易日历太短，请检查 --calendar-start")

    info.update({
        "first_date": str(idx.min().date()),
        "last_date": str(idx.max().date()),
        "days": int(len(idx)),
    })
    return idx, info


def load_policies(path):
    policies = json.loads(json.dumps(DEFAULT_POLICIES))
    if path:
        updates = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, update in updates.items():
            if key not in DEFAULT_POLICIES and not key.startswith("macro:"):
                raise ValueError(f"未知策略键：{key}")
            base = policies.get(key, policies["macro"]).copy()
            base.update(update)
            policies[key] = base

    for key, policy in policies.items():
        if set(policy) != {"lag_days", "release_hour", "max_age_days"}:
            raise ValueError(f"{key} 策略字段不正确")
        if policy["lag_days"] < 0 or policy["max_age_days"] < 0:
            raise ValueError(f"{key} 的滞后/年龄不能为负")
        if not 0 <= policy["release_hour"] <= 23:
            raise ValueError(f"{key} release_hour 非法")
        # 防泄露护栏：本库全部来源都是日频观测（行情/估值/两融收盘后才有，
        # 百度指数更晚），观测日 d 的值最早在 d 收盘（约 15:00）后才存在。
        # lag_days=0 且 release_hour<15 等于声称"当日盘中值已知"，会把
        # 尚未发生的值当作可用（例如读取到 next_session 的日线）。
        if policy["lag_days"] == 0 and policy["release_hour"] < 15:
            raise ValueError(
                f"{key}：lag_days=0 且 release_hour={policy['release_hour']} "
                "不合理——日频观测值最早在观测日收盘后才存在，该配置会把"
                "未发生的值当作可用（未来数据泄露）。请增大 lag_days 或把"
                "release_hour 设为不早于实际发布时刻的小时（>=15）。"
            )
    return policies


def select_stocks(con, symbol=None, industry=None, top=None):
    expr = sql_symbol("symbol")
    conditions, params = [], []

    if symbol:
        conditions.append(f"{expr} = ?")
        params.append(norm_symbol(symbol))
    if industry:
        conditions.append("industry_l2 = ?")
        params.append(industry)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    df = con.execute(
        f"""
        SELECT {expr} AS symbol, name, exchange, industry_l2, total_mv
        FROM stock_list
        {where}
        ORDER BY total_mv DESC NULLS LAST
        """,
        params,
    ).fetchdf()

    assert_unique(df, ["symbol"], "stock_list 标准化代码")
    if df.empty:
        raise ValueError("没有符合条件的股票")

    for value in df["symbol"]:
        norm_symbol(value)

    if top is not None:
        df = df.head(top)
    elif not symbol and not industry:
        df = df.head(1)

    return df


def load_prices(con, symbol, start, end):
    df = con.execute(
        f"""
        SELECT date, open, close, high, low, volume, turnover
        FROM akshare_tx
        WHERE {sql_symbol("symbol")} = ?
          AND adjust = 'hfq'
          AND date >= ? AND date <= ?
        ORDER BY date
        """,
        [symbol, start.date(), end.date()],
    ).fetchdf()

    if df.empty:
        raise ValueError(f"{symbol} 没有指定区间的后复权行情")

    df = parse_dates(df)
    assert_unique(df, ["date"], f"行情/{symbol}")
    return as_numeric(df, PRICE_FIELDS).set_index("date").sort_index()


def empty_series():
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def make_specs_from_long(df, key_col, keys, prefix, source, transform):
    df = parse_dates(df)
    assert_unique(df, ["date", key_col], source)
    df = as_numeric(df, ["value"])

    specs = []
    for key in keys:
        part = df[df[key_col] == key]
        s = (
            part.set_index("date")["value"].sort_index()
            if not part.empty else empty_series()
        )
        specs.append({
            "column": f"{prefix}{key}",
            "source": source,
            "policy_key": source,
            "transform": transform,
            "series": s,
        })
    return specs


def load_stock_aux(con, symbol, keyword, start, end):
    bounds = [start.date(), end.date()]

    baidu = con.execute(
        """
        SELECT date, source, value FROM baidu
        WHERE keyword = ?
          AND source IN ('search_all', 'feed')
          AND date >= ? AND date <= ?
        """,
        [keyword] + bounds,
    ).fetchdf()

    val = con.execute(
        f"""
        SELECT date, indicator, value FROM guzhi
        WHERE {sql_symbol("symbol")} = ?
          AND indicator IN ('pe', 'pb', 'ps', 'pcf')
          AND date >= ? AND date <= ?
        """,
        [symbol] + bounds,
    ).fetchdf()

    rzrq = con.execute(
        f"""
        SELECT date, fin_tval_ratio, loan_tval_ratio FROM rzrq
        WHERE {sql_symbol("code")} = ?
          AND date >= ? AND date <= ?
        """,
        [symbol] + bounds,
    ).fetchdf()

    specs = make_specs_from_long(
        baidu, "source", BAIDU_FIELDS, "baidu_", "baidu", "log1p"
    )
    specs += make_specs_from_long(
        val, "indicator", VAL_FIELDS, "val_", "guzhi", "signed_log1p"
    )

    rzrq = parse_dates(rzrq)
    assert_unique(rzrq, ["date"], f"rzrq/{symbol}")
    rzrq = as_numeric(rzrq, RZRQ_FIELDS)
    for col in RZRQ_FIELDS:
        s = (
            rzrq.set_index("date")[col].sort_index() / 100.0
            if not rzrq.empty else empty_series()
        )
        specs.append({
            "column": f"rzrq_{col}",
            "source": "rzrq",
            "policy_key": "rzrq",
            "transform": "identity",
            "series": s,
        })
    return specs


def macro_column(symbol, field):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")
    digest = hashlib.sha256(symbol.encode()).hexdigest()[:8]
    return f"macro_{field}_{safe}_{digest}"


def load_macro(con, fields, symbols, start, end):
    """返回宏观行情规格列表。

    symbols 语义：None 表示自动发现区间内全部品种；显式空列表表示
    "确实没有宏观品种"，直接返回空（不会被误当作自动发现）。
    """
    if not set(fields).issubset({"close", "volume"}):
        raise ValueError("宏观字段只能为 close、volume")
    if symbols is not None and len(symbols) == 0:
        return []

    params = [start.date(), end.date()]
    where = "date >= ? AND date <= ?"
    if symbols is not None:
        where += " AND symbol IN (" + ",".join("?" for _ in symbols) + ")"
        params.extend(symbols)

    # fields 已经过白名单校验。
    df = con.execute(
        f"SELECT symbol, date, {', '.join(fields)} "
        f"FROM tradingview_macro WHERE {where}",
        params,
    ).fetchdf()

    df = parse_dates(df)
    assert_unique(df, ["symbol", "date"], "tradingview_macro")
    df = as_numeric(df, fields)

    if symbols is not None:
        selected = sorted(set(symbols))
    else:
        selected = sorted(df["symbol"].dropna().unique().tolist())

    specs = []
    for symbol in selected:
        part = df[df["symbol"] == symbol]
        for field in fields:
            s = (
                part.set_index("date")[field].sort_index()
                if not part.empty else empty_series()
            )
            specs.append({
                "column": macro_column(symbol, field),
                "source": "macro",
                "policy_key": f"macro:{symbol}:{field}",
                "transform": "log1p" if field == "volume" else "identity",
                "symbol": symbol,
                "field": field,
                "series": s,
            })
    return specs


def align_one(spec, decision_times, policies):
    """按假设可用时间匹配最新非空值，过期则置空。"""
    col = spec["column"]
    policy = policies.get(spec["policy_key"], policies[spec["source"]])
    series = spec["series"].dropna().sort_index()
    series = series[np.isfinite(series.to_numpy(dtype=float))]

    if spec["transform"] == "log1p" and (series < 0).any():
        raise ValueError(f"{col} 存在不允许的负值")

    index = decision_times.index
    if series.empty:
        return (
            pd.Series(np.nan, index=index),
            pd.Series(np.nan, index=index),
            pd.DataFrame({
                f"{col}__observed": pd.NaT,
                f"{col}__available": pd.NaT,
            }, index=index),
        )

    right = pd.DataFrame({
        "observed": series.index,
        "value": series.to_numpy(dtype=float),
    })
    right["available"] = (
        right["observed"]
        + pd.Timedelta(days=policy["lag_days"])
        + pd.Timedelta(hours=policy["release_hour"])
    )

    matched = pd.merge_asof(
        pd.DataFrame({"decision": decision_times.to_numpy()}),
        right.sort_values("available"),
        left_on="decision",
        right_on="available",
        direction="backward",
    )

    age = (
        matched["decision"] - matched["observed"]
    ).dt.total_seconds() / 86400.0

    usable = (
        matched["value"].notna()
        & age.ge(0)
        & age.le(policy["max_age_days"])
    )
    values = pd.Series(
        matched["value"].where(usable).to_numpy(), index=index
    )
    ages = pd.Series(age.to_numpy(), index=index)
    audit = pd.DataFrame({
        f"{col}__observed": matched["observed"].to_numpy(),
        f"{col}__available": matched["available"].to_numpy(),
    }, index=index)

    known = matched["available"].notna()
    if (
        matched.loc[known, "available"] > matched.loc[known, "decision"]
    ).any():
        raise AssertionError("发现未来可用数据")

    return values, ages, audit


def fixed_transform(values, kind):
    if kind == "log1p":
        return np.log1p(values)
    if kind == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    return values


def build_features(prices, calendar, as_of, specs, config):
    if as_of not in calendar:
        raise ValueError("as-of 不在交易日历中")
    pos = calendar.get_loc(as_of)
    if pos + 1 >= len(calendar):
        raise ValueError("日历必须包含 as-of 之后的下一交易日")

    unknown = prices.index.difference(calendar)
    if len(unknown):
        raise ValueError(f"行情日期不在交易日历中：{unknown[:5].tolist()}")

    index = calendar[
        (calendar >= prices.index.min()) & (calendar <= as_of)
    ]
    p = prices.reindex(index).copy()

    next_date = pd.Series(calendar[1:].to_numpy(), index=calendar[:-1])
    decision = (
        next_date.reindex(index)
        + pd.Timedelta(
            hours=config["decision_hour"],
            minutes=config["decision_minute"],
        )
    )

    ohlc = p[["open", "close", "high", "low"]]
    valid_price = (
        ohlc.notna().all(axis=1)
        & ohlc.gt(0).all(axis=1)
        & p["high"].ge(p["low"])
        & p["high"].ge(p[["open", "close"]].max(axis=1))
        & p["low"].le(p[["open", "close"]].min(axis=1))
    )
    p.loc[~valid_price, ["open", "close", "high", "low"]] = np.nan
    p.loc[p["turnover"] < 0, "turnover"] = np.nan
    # 成交量缺失/为负只令特征缺失，不单独判定停牌（零量不等于停牌，
    # 停牌日在腾讯行情中通常整行缺失，由 core_valid 机制处理）。
    p.loc[p["volume"] < 0, "volume"] = np.nan

    prev = p["close"].shift(1)
    simple_return = p["close"] / prev - 1.0
    target = (
        np.log(p["close"] / prev)
        if config["return_type"] == "log"
        else simple_return
    )

    raw = p.rename(columns={c: f"px_{c}" for c in PRICE_FIELDS})

    # 行情派生特征。逐列插入会导致 DataFrame 碎片化，先收集再一次拼接。
    derived = {
        "px_open_gap": p["open"] / prev - 1.0,
        "px_intraday_return": p["close"] / p["open"] - 1.0,
        "px_amplitude": (p["high"] - p["low"]) / prev,
    }
    for window in (5, 20):
        derived[f"px_return_{window}"] = (
            (1 + simple_return)
            .rolling(window, min_periods=window)
            .apply(np.prod, raw=True) - 1
        )
        derived[f"px_volatility_{window}"] = (
            simple_return.rolling(window).std()
        )
        derived[f"px_ma_ratio_{window}"] = (
            p["close"] / p["close"].rolling(window).mean() - 1
        )

    mean_turnover = p["turnover"].rolling(20).mean().replace(0, np.nan)
    derived["px_turnover_relative_20"] = p["turnover"] / mean_turnover - 1

    # groups 用于把数值、缺失标记、数据年龄作为一组筛选。
    groups = {}
    base_columns = {c: raw[c] for c in raw.columns}
    base_columns.update(derived)
    new_columns = dict(derived)
    for col in base_columns:
        new_columns[missing_col(col)] = (
            base_columns[col].isna().astype(float)
        )
        groups[col] = [col, missing_col(col)]

    features = pd.concat(
        [raw, pd.DataFrame(new_columns, index=index)], axis=1
    )
    # 成交量原值单位为股、量级巨大（1e8~1e11），特征侧用 log1p；
    # aligned_values.csv 中 px_volume 仍保留原值。
    features["px_volume"] = np.log1p(features["px_volume"])

    # 目标日星期 one-hot（周一~周五）：行 t 的目标日 = 以该行为输入日
    # 时的预测目标，即下一交易日（decision 的日期部分）。星期是纯日历
    # 信息，预测时点完全已知，不引入未来数据；样本窗口 [start..end] 的
    # 目标日 date[end+1] 的 one-hot 恰为窗口最后一行（决策行），窗口内
    # 其余各行各自表示"以该行为输入日"的目标日星期（窗口互相重叠，
    # 行级存储无法整窗恒定）。交易日只落在周一~周五，5 列完整覆盖；
    # 恒不缺失，不生成 __missing / __age_days 伴随列，也不参与屏蔽。
    target_dow = next_date.reindex(index).dt.dayofweek
    for i, name in enumerate(TARGET_DOW_NAMES):
        col = dow_column(name)
        features[col] = (target_dow == i).astype(float)
        groups[col] = [col]

    # 辅助来源按各自的可用时间对齐，禁止 bfill。
    audit = pd.DataFrame({"decision_time": decision}, index=index)
    raw_aux, aux_columns, audits = {}, {}, []
    for spec in specs:
        col = spec["column"]
        values, age, part_audit = align_one(
            spec, decision, config["policies"]
        )
        raw_aux[col] = values
        aux_columns[col] = fixed_transform(values, spec["transform"])
        aux_columns[age_col(col)] = age
        aux_columns[missing_col(col)] = values.isna().astype(float)
        groups[col] = [
            col, age_col(col), missing_col(col)
        ]
        audits.append(part_audit)

    if raw_aux:
        raw = pd.concat([raw, pd.DataFrame(raw_aux, index=index)], axis=1)
        features = pd.concat(
            [features, pd.DataFrame(aux_columns, index=index)], axis=1
        )
        audit = pd.concat([audit] + audits, axis=1)

    raw[TARGET] = target
    features[TARGET] = target
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features[[c for c in features if c != TARGET] + [TARGET]]

    # 停牌/无行情日及其后无法计算单日收益率的日期不作为有效核心行。
    core_valid = valid_price & target.notna()
    return raw, features, audit, core_valid, valid_price, groups


def make_samples(features, core_valid, config, train_end, val_end,
                 start=None, tail=None, split_layout="train_val_test"):
    """返回 (样本表, 候选目标日数, 剔除统计)。

    split_layout 见模块常量 SPLIT_LAYOUTS：
      train_val_test：target <= train_end 为 train，<= val_end 为 val，
        其余为 test；
      train_test：target <= train_end（此时 train_end 即重拟合截止日
        fit_end，包含原验证区间）为 train，其余为 test，不产生 val 样本。

    样本口径区分"可预测 / 可评分"：
      输入窗口有效的行全部保留（该预测在目标日早 08:30 本可合法发出），
      其中目标日无有效行情或标签的行标记 label_valid=False、y_raw 为空，
      保留在 samples.csv 中作为覆盖登记；监督训练/评估装载时只取
      label_valid=True 的行（见 fit_preprocessing / PanelWindows）。
      评分时排除无标签样本不可避免，但不得把它伪装成"那天没有生成预测"。

    剔除统计区分三类：
      insufficient_history：历史不足，尚不能开窗（结构性预热）；
      invalid_input_window：输入窗口含无效核心行（多为停牌），无法在
        预测时点生成输入；
      invalid_target_label：输入窗口有效、预测本可发出，但目标日无有效
        行情或标签（= samples 中 label_valid=False 的行数，事后筛选
        偏差应被报告而非静默）。
    """
    if split_layout not in SPLIT_LAYOUTS:
        raise ValueError(f"不支持的 split_layout：{split_layout}")
    if split_layout == "train_test" and val_end is not None:
        raise ValueError("train_test 布局没有验证集，val_end 必须为 None")
    length = config["seq_len"]
    index = features.index
    y = features[TARGET].to_numpy(dtype=float)
    valid = core_valid.to_numpy(dtype=bool)

    # 目标日期限制，不先截断底层行情，保留滚动计算预热。
    candidate_targets = index[1:]
    if start is not None:
        candidate_targets = candidate_targets[candidate_targets >= start]
    if tail:
        candidate_targets = candidate_targets[-tail:]
    allowed = set(candidate_targets)

    records = []
    # 前缀和：O(1) 检查一个窗口是否存在无效核心行。
    bad_prefix = np.r_[0, np.cumsum(~valid)]
    exclusions = {
        "insufficient_history": 0,
        "invalid_input_window": 0,
        "invalid_target_label": 0,
    }
    positions = index.get_indexer(candidate_targets)
    exclusions["insufficient_history"] = int((positions < length).sum())

    for end in range(length - 1, len(index) - 1):
        first, target_pos = end - length + 1, end + 1
        target_date = index[target_pos]
        if target_date not in allowed:
            continue
        if bad_prefix[end + 1] - bad_prefix[first] != 0:
            exclusions["invalid_input_window"] += 1
            continue

        label_valid = bool(
            valid[target_pos] and np.isfinite(y[target_pos])
        )
        if not label_valid:
            exclusions["invalid_target_label"] += 1

        if split_layout == "train_test":
            split = "train" if target_date <= train_end else "test"
        else:
            split = (
                "train" if target_date <= train_end
                else "val" if target_date <= val_end
                else "test"
            )
        records.append({
            "start_pos": first,
            "end_pos": end,
            "target_pos": target_pos,
            "input_date": index[end],
            "target_date": target_date,
            "split": split,
            "label_valid": label_valid,
            "y_raw": float(y[target_pos]) if label_valid else np.nan,
        })

    columns = [
        "start_pos", "end_pos", "target_pos", "input_date",
        "target_date", "split", "label_valid", "y_raw",
    ]
    result = pd.DataFrame(records, columns=columns)
    return result, len(candidate_targets), exclusions


def fit_preprocessing(features, samples, groups, min_coverage, min_obs):
    # 只用合法训练样本（label_valid）拟合：无标签的"可预测"行不参与
    # 统计量估计，也不贡献 fit 行。
    train = samples[
        (samples["split"] == "train") & samples["label_valid"]
    ]
    if train.empty:
        raise ValueError("没有有效训练样本")

    # 差分数组构造 fit mask：O(样本数 + 行数)，避免逐窗口切片赋值。
    diff = np.zeros(len(features) + 1, dtype=np.int64)
    starts = train["start_pos"].to_numpy(dtype=np.int64)
    ends = train["end_pos"].to_numpy(dtype=np.int64) + 1
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    fit_mask = np.cumsum(diff[:-1]) > 0
    fit = features.loc[fit_mask]

    keep, dropped, coverage = [], {}, {}
    for base, columns in groups.items():
        series = fit[base]
        cov = float(series.notna().mean())
        coverage[base] = cov

        if cov < min_coverage or series.notna().sum() < min_obs:
            dropped[base] = "low_train_coverage_or_observations"
        elif series.dropna().nunique() <= 1:
            dropped[base] = "constant_in_train"
        else:
            keep.extend(columns)

    # 目标历史列必须保留，且置于最后。
    keep.append(TARGET)
    if len(keep) < 2:
        raise ValueError("没有足够的可用特征")

    stats = {}
    for col in keep:
        values = fit[col].dropna()
        if col == TARGET and len(values) < min_obs:
            raise ValueError("训练收益率有效值不足")
        if values.empty:
            raise ValueError(f"保留列 {col} 在训练集全空")

        if is_missing_col(col):
            # 二值标记保持 0/1。
            stats[col] = {"median": 1.0, "mean": 0.0, "scale": 1.0}
            continue

        if is_dow_col(col):
            # 目标日星期 one-hot 保持 0/1，不标准化。
            stats[col] = {"median": 0.0, "mean": 0.0, "scale": 1.0}
            continue

        median = float(values.median())
        filled = fit[col].fillna(median)
        mean = float(filled.mean())
        scale = float(filled.std(ddof=0))

        if not np.isfinite(scale) or scale < 1e-12:
            if col == TARGET:
                raise ValueError("训练收益率为常数")
            scale = 1.0

        stats[col] = {"median": median, "mean": mean, "scale": scale}

    return {
        "feature_columns": keep,
        "target": TARGET,
        "statistics": stats,
        "dropped_groups": dropped,
        "train_group_coverage": coverage,
        "fit_row_count": int(fit_mask.sum()),
        "fit_first_date": str(fit.index.min().date()),
        "fit_last_date": str(fit.index.max().date()),
    }


def transform(features, preprocessing):
    columns = preprocessing["feature_columns"]
    missing = set(columns) - set(features)
    if missing:
        raise ValueError(f"推理数据缺少训练特征定义：{sorted(missing)}")

    result = np.empty((len(features), len(columns)), dtype=np.float32)
    for i, col in enumerate(columns):
        p = preprocessing["statistics"][col]
        values = (
            (features[col].fillna(p["median"]) - p["mean"]) / p["scale"]
        )
        result[:, i] = values.to_numpy(dtype=np.float32)

    if not np.isfinite(result).all():
        raise ValueError("最终矩阵存在 NaN、无穷值或 float32 溢出")
    return result


def save_latest(out, matrix, features, core_valid, calendar, config):
    length = config["seq_len"]
    if len(matrix) < length or not core_valid.iloc[-length:].all():
        return {
            "saved": False,
            "reason": "latest_window_too_short_or_invalid",
        }

    input_date = features.index[-1]
    target_date = calendar[calendar.get_loc(input_date) + 1]
    np.savez_compressed(
        out / "latest.npz",
        X=matrix[-length:][None, ...],
        input_date=np.asarray([input_date], dtype="datetime64[D]"),
        target_date=np.asarray([target_date], dtype="datetime64[D]"),
    )
    return {
        "saved": True,
        "input_date": str(input_date.date()),
        "target_date": str(target_date.date()),
    }


def load_asof(args, con, calendar):
    if args.as_of:
        as_of = pd.Timestamp(args.as_of).normalize()
    else:
        value = con.execute(
            "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
        ).fetchone()[0]
        if value is None:
            raise ValueError("数据库没有后复权行情")
        as_of = pd.Timestamp(value).normalize()

    if as_of not in calendar:
        raise ValueError("as-of 不在交易日历内")
    if calendar.get_loc(as_of) + 1 >= len(calendar):
        raise ValueError("交易日历缺少下一交易日")
    return as_of


def coverage_report(features, valid_price, samples, train_end, val_end):
    index = features.index
    # train_test 布局没有验证区间：train <= train_end（即 fit_end），
    # test 直接从 train_end 之后开始。
    if val_end is None:
        masks = {
            "train": index <= train_end,
            "test": index > train_end,
        }
    else:
        masks = {
            "train": index <= train_end,
            "val": (index > train_end) & (index <= val_end),
            "test": index > val_end,
        }
    report = {}
    for name, mask in masks.items():
        if not mask.any():
            report[name] = {"rows": 0}
            continue
        part = features.loc[mask]
        in_split = samples["split"] == name
        report[name] = {
            "rows": int(mask.sum()),
            "valid_price_ratio": float(valid_price.loc[mask].mean()),
            # 可评分样本（label_valid）；可预测行数见 eligible_signals。
            "samples": int((in_split & samples["label_valid"]).sum()),
            "eligible_signals": int(in_split.sum()),
            "feature_nonmissing_ratio": {
                c: float(part[c].notna().mean()) for c in part
            },
        }
    return report


def calendar_fingerprint(calendar, as_of, info):
    """训练期交易日历指纹：infer 时冻结日历参数并校验历史前缀一致。"""
    dates = calendar[calendar <= as_of]
    payload = "\n".join(d.strftime("%Y-%m-%d") for d in dates)
    return {
        "source": info.get("source"),
        "path": info.get("path"),
        "sha256_of_calendar_csv": info.get("sha256"),
        "calendar_start": info.get("calendar_start"),
        "min_market_symbols": info.get("min_market_symbols"),
        "as_of": str(as_of.date()),
        "trading_days_up_to_as_of": int(len(dates)),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def calendar_dates_sha256(calendar, as_of):
    dates = calendar[calendar <= pd.Timestamp(as_of)]
    payload = "\n".join(d.strftime("%Y-%m-%d") for d in dates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(dates)


def source_watermarks(con, symbol, keyword):
    """各来源当前入库的最大观测日期（信息水位，不代表真实发布时间）。"""

    def q(sql, params):
        value = con.execute(sql, params).fetchone()
        if not value or value[0] is None:
            return None
        return str(pd.Timestamp(value[0]).date())

    return {
        "akshare_tx": q(
            f"SELECT max(date) FROM akshare_tx "
            f"WHERE {sql_symbol('symbol')} = ? AND adjust = 'hfq'",
            [symbol],
        ),
        "baidu": q(
            "SELECT max(date) FROM baidu WHERE keyword = ?", [keyword]
        ),
        "guzhi": q(
            f"SELECT max(date) FROM guzhi WHERE {sql_symbol('symbol')} = ?",
            [symbol],
        ),
        "rzrq": q(
            f"SELECT max(date) FROM rzrq WHERE {sql_symbol('code')} = ?",
            [symbol],
        ),
        "tradingview_macro": q(
            "SELECT max(date) FROM tradingview_macro", []
        ),
    }


def history_drift_report(preprocessor_path, matrix, features, columns):
    """比对 infer 重算的历史矩阵与训练期导出矩阵。

    infer 只冻结预处理参数，历史行按当前数据库重算；若数据库历史值被
    修订（如 TradingView 全历史重抓覆盖），相同日期的特征行会变化。
    返回逐行漂移统计，供 manifest 与告警使用。

    columns 为矩阵的列名（= 预处理 feature_columns）；features.columns
    包含被覆盖率过滤掉的列，不能用于矩阵列名映射。
    """
    base = Path(preprocessor_path).parent
    matrix_path, dates_path = base / "feature_matrix.npy", base / "dates.npy"
    if not (matrix_path.exists() and dates_path.exists()):
        return {"checked": False, "reason": "训练目录缺少矩阵文件，无法比对"}

    old_matrix = np.load(matrix_path)
    old_dates = pd.DatetimeIndex(np.load(dates_path))
    if old_matrix.shape[1] != matrix.shape[1] or len(columns) != matrix.shape[1]:
        return {
            "checked": False,
            "reason": (
                f"特征列数不同：训练期 {old_matrix.shape[1]} vs 当前 "
                f"{matrix.shape[1]}"
            ),
        }

    new_dates = features.index
    common = old_dates.intersection(new_dates)
    if len(common) == 0:
        return {"checked": False, "reason": "无重叠日期"}

    old_pos = old_dates.get_indexer(common)
    new_pos = new_dates.get_indexer(common)
    delta = np.abs(
        old_matrix[old_pos].astype(np.float64) - matrix[new_pos].astype(np.float64)
    )
    changed_rows = np.flatnonzero((delta > 1e-9).any(axis=1))
    changed_columns = [
        columns[i] for i in np.flatnonzero((delta > 1e-9).any(axis=0))
    ]
    return {
        "checked": True,
        "compared_rows": int(len(common)),
        "changed_rows": int(len(changed_rows)),
        "changed_row_dates": [
            str(new_dates[new_pos[i]].date()) for i in changed_rows[:20]
        ],
        "changed_columns": changed_columns,
        "max_abs_diff": float(delta.max()) if delta.size else 0.0,
    }


def prepare_one(con, row, args, calendar, as_of, macro_specs, config,
                calendar_info, frozen=None):
    symbol = row["symbol"]
    config = json.loads(json.dumps(config))
    config["baidu_keyword"] = config.get("baidu_keyword") or str(row["name"])

    # 保留整个日历覆盖区间用于预热。
    # 辅助来源额外向前读取，以便日历第一行仍能匹配最近可用观测。
    max_lookback = max(
        p["max_age_days"] + p["lag_days"] for p in config["policies"].values()
    )
    aux_start = calendar.min() - pd.Timedelta(days=max_lookback + 10)
    next_session = calendar[calendar.get_loc(as_of) + 1]

    prices = load_prices(con, symbol, calendar.min(), as_of)
    specs = load_stock_aux(
        con, symbol, config["baidu_keyword"], aux_start, next_session
    ) + macro_specs

    raw, features, audit, core_valid, valid_price, groups = build_features(
        prices, calendar, as_of, specs, config
    )

    if frozen is None:
        split_layout = getattr(args, "split_layout", None) or "train_val_test"
        if split_layout not in SPLIT_LAYOUTS:
            raise ValueError(f"不支持的 split_layout：{split_layout}")

        train_end = pd.Timestamp(args.train_end).normalize()
        if split_layout == "train_val_test":
            if args.val_end is None:
                raise ValueError("train_val_test 布局必须提供 val_end")
            val_end = pd.Timestamp(args.val_end).normalize()
        else:
            if args.val_end is not None:
                raise ValueError(
                    "train_test 布局没有验证集，不能提供 val_end"
                )
            val_end = None
        start = pd.Timestamp(args.start).normalize() if args.start else None

        samples, candidates, exclusions = make_samples(
            features, core_valid, config, train_end, val_end,
            start=start, tail=args.tail, split_layout=split_layout,
        )
        # 训练资格只由训练期（且标签有效）的样本量决定；验证/测试覆盖率
        # 是评估信息，不决定训练资格（否则未来评估期停牌/缺数/退市会
        # 反过来剔除训练完全合格的股票，形成测试期筛选偏差）。
        scoreable = samples[samples["label_valid"]]
        counts = scoreable["split"].value_counts()
        if int(counts.get("train", 0)) < args.min_train_samples:
            raise ValueError(
                f"train 样本不足：{int(counts.get('train', 0))} "
                f"< {args.min_train_samples}"
            )

        preprocessing = fit_preprocessing(
            features, samples, groups,
            args.min_feature_coverage, args.min_observations,
        )
        preprocessing.update({
            "version": VERSION,
            "symbol": symbol,
            "config": config,
            "split_layout": split_layout,
            "train_end": args.train_end,
            "val_end": args.val_end,
            # 样本选择口径登记：--start 限制最早目标日期（walk-forward
            # rolling 每折不同），--tail 限制最近候选目标日；两者都改变
            # 样本集合，导出复用与训练装载时都会比对，防止不同训练起点
            # 的导出被误复用。
            "sample_selection": {
                "start": str(start.date()) if start is not None else None,
                "tail": args.tail,
            },
            # validate 模式重拟合对比需要与训练一致的过滤阈值。
            "min_feature_coverage": args.min_feature_coverage,
            "min_observations": args.min_observations,
            # 日历指纹：infer 冻结日历参数并校验历史前缀。
            "calendar": calendar_fingerprint(
                calendar, as_of, calendar_info
            ),
        })

        report = coverage_report(
            features, valid_price, samples, train_end, val_end
        )
        # 验证/测试覆盖率登记（两种布局一致）：单股下限只用于报告对照，
        # 不决定训练资格。整体层面"该折是否有足够验证证据选模型"由
        # 训练装载检查（train_global_timexer）负责。
        report["eval_sample_minimum"] = {
            "required_per_stock": args.min_eval_samples,
            "enforced": False,
            "note": (
                "验证/测试覆盖率为评估信息，不决定训练资格；单股仅强制"
                "训练样本下限，评估期按实际有效样本参与"
            ),
            "per_stock_scoreable": {
                "train": int(counts.get("train", 0)),
                **(
                    {"val": int(counts.get("val", 0))}
                    if split_layout == "train_val_test" else {}
                ),
                "test": int(counts.get("test", 0)),
            },
        }
        report["candidate_target_dates"] = candidates
        # valid_samples 为可评分样本（label_valid）；eligible_samples 为
        # 可预测信号（含目标日无标签的行）。
        report["valid_samples"] = int(len(scoreable))
        report["eligible_samples"] = int(len(samples))
        report["valid_sample_ratio"] = (
            float(len(scoreable) / candidates) if candidates else 0.0
        )
        report["sample_exclusions"] = exclusions
        report["prediction_eligible_dates"] = (
            candidates
            - exclusions["insufficient_history"]
            - exclusions["invalid_input_window"]
        )
    else:
        preprocessing = frozen
        samples = None
        report = {
            "mode": "infer",
            "rows": len(features),
            f"latest_{config['seq_len']}_valid_core_rows": int(
                core_valid.iloc[-config["seq_len"]:].sum()
            ),
        }

    matrix = transform(features, preprocessing)

    warnings = []
    # 最新输入行的辅助特征缺失/过期（数据未到或超出 max_age）。
    aux_bases = [
        c for c in features.columns
        if is_exogenous_col(c) and not is_companion_col(c)
    ]
    stale_latest = [
        c for c in aux_bases if not np.isfinite(features[c].iloc[-1])
    ]
    if stale_latest:
        warnings.append(
            "最新输入行辅助特征缺失或过期（数据未到或超出 max_age）："
            + ", ".join(stale_latest[:10])
            + (" 等" if len(stale_latest) > 10 else "")
        )
    removed_days = calendar_info.get("removed_no_quote_days") or []
    if removed_days:
        warnings.append(
            "从日历剔除的无行情推导交易日："
            + ", ".join(d["date"] for d in removed_days)
            + "（如属采集故障而非休市，请改用权威 --calendar）"
        )
    partial_days = calendar_info.get("partial_coverage_days") or []
    if partial_days:
        warnings.append(
            f"存在 {len(partial_days)} 个行情覆盖不足的交易日，"
            "已保留在日历并标记为数据异常"
        )

    history_drift = None
    if frozen is not None:
        history_drift = history_drift_report(
            args.preprocessor, matrix, features,
            preprocessing["feature_columns"],
        )
        report["history_drift"] = history_drift
        if history_drift.get("checked") and history_drift["changed_rows"] > 0:
            msg = (
                f"历史特征行与训练期不一致：{history_drift['changed_rows']}/"
                f"{history_drift['compared_rows']} 行变化，max|Δ|="
                f"{history_drift['max_abs_diff']:.3g}（数据库历史数据可能被修订）"
            )
            warnings.append(msg)
            print(f"[WARN] {symbol} {msg}")
            if args.fail_on_history_drift:
                raise ValueError(msg + "；已按 --fail-on-history-drift 失败")

    if samples is not None:
        p = preprocessing["statistics"][TARGET]
        samples["y_scaled"] = (samples["y_raw"] - p["mean"]) / p["scale"]

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / symbol
    if final.exists():
        raise FileExistsError(
            f"输出目录已存在：{final}。请换一个新的 --out-dir，避免混用版本。"
        )

    temp = Path(tempfile.mkdtemp(prefix=f".{symbol}_", dir=root))
    try:
        raw.to_csv(temp / "aligned_values.csv", index_label="date")
        features.to_csv(
            temp / "features_unscaled.csv", index_label="date"
        )
        audit.to_csv(temp / "alignment_audit.csv", index_label="date")
        np.save(temp / "feature_matrix.npy", matrix)
        np.save(
            temp / "dates.npy",
            features.index.to_numpy(dtype="datetime64[D]"),
        )

        if samples is not None:
            samples.to_csv(temp / "samples.csv", index=False)
            # 训练前数据预览：逐交易日的标准化特征（与 feature_matrix.npy
            # 逐值一致，即模型实际吃到的数值）+ 当日原始收益率 + 按目标
            # 日期口径的切分标签（与 samples.csv 的 split 规则一致）。
            preview = pd.DataFrame(
                matrix, index=features.index,
                columns=preprocessing["feature_columns"],
            )
            preview["y_raw"] = features[TARGET].to_numpy(dtype=float)
            if split_layout == "train_test":
                preview["split"] = np.where(
                    features.index <= train_end, "train", "test"
                )
            else:
                preview["split"] = np.select(
                    [features.index <= train_end, features.index <= val_end],
                    ["train", "val"], default="test",
                )
            preview.to_csv(temp / "train_preview.csv", index_label="date")

        latest = save_latest(
            temp, matrix, features, core_valid, calendar, config
        )
        json_write(temp / "preprocessing.json", preprocessing)

        manifest = {
            "version": VERSION,
            "mode": args.mode,
            "created_at": datetime.now().astimezone().isoformat(),
            "symbol": symbol,
            "name": None if pd.isna(row["name"]) else str(row["name"]),
            "industry_snapshot": (
                None if pd.isna(row.get("industry_l2"))
                else str(row["industry_l2"])
            ),
            "as_of": str(as_of.date()),
            # 预测时点：下一交易日 decision_hour:minute；导出时点未必等于
            # 该时刻（例如周五收盘后导出的是"周一 08:30"的窗口，周末及
            # 周一早上的信息尚未到齐），由 source_watermarks 反映数据水位。
            "forecast_time": (
                next_session
                + pd.Timedelta(
                    hours=config["decision_hour"],
                    minutes=config["decision_minute"],
                )
            ).isoformat(),
            "source_watermarks": source_watermarks(
                con, symbol, config["baidu_keyword"]
            ),
            "warnings": warnings,
            "seq_len": config["seq_len"],
            "pred_len": 1,
            "return_type": config["return_type"],
            "target": TARGET,
            "enc_in": matrix.shape[1],
            "matrix_shape": list(matrix.shape),
            "latest": latest,
            "quality": report,
            "config": config,
            "db_path": str(Path(args.db).resolve()),
            "calendar": calendar_info,
            "macro_mapping": [
                {k: v for k, v in spec.items() if k != "series"}
                for spec in macro_specs
            ],
            "limitations": [
                "原表没有实际发布时间，采用配置的可用时间假设；"
                "source_watermarks 只是入库水位，不能证明 point-in-time。",
                "历史复权、财务和宏观数据可能存在事后修订；"
                "infer 的 history_drift 只能发现、不能消除该问题。",
                "默认当前股票简称匹配百度，未自动恢复历史更名。",
                "当前行业、市值和存续名单只用于选股，存在样本选择局限。",
                "交易日历只覆盖 --calendar-start 之后的历史。",
                "数据库推导的日历以行情数据校准，未来交易日来自"
                "holiday_calendar，可能存在误差。",
                "未自动识别接口以平价零量记录表示的停牌，需要来源核验。",
                "预测的是统计口径收益率，不等于可实现交易收益"
                "（t+1 早 08:30 出信号后无法按 t 收盘价成交）。",
            ],
        }
        if frozen is not None:
            manifest["frozen_preprocessing_sha256"] = sha256_file(
                args.preprocessor
            )
        manifest["artifact_sha256"] = {
            p.name: sha256_file(p) for p in sorted(temp.iterdir())
        }
        json_write(temp / "manifest.json", manifest)
        os.replace(temp, final)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    count_text = ""
    if samples is not None:
        scoreable_counts = samples.loc[
            samples["label_valid"], "split"
        ].value_counts().to_dict()
        unscoreable = int((~samples["label_valid"]).sum())
        count_text = (
            str(scoreable_counts)
            + (f" | unscoreable={unscoreable}" if unscoreable else "")
        )
    print(
        f"[OK] {symbol} {row['name']} | "
        f"F={matrix.shape[1]} | {count_text} | latest={latest['saved']}"
        + (f" | warnings={len(warnings)}" if warnings else "")
    )
    return {"symbol": symbol, "status": "ok", "directory": str(final)}


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


def _stats_close(a, b, atol=1e-9):
    """比较两份 preprocessing 统计量（容差比较，避免浮点误报）。"""
    if set(a) != set(b):
        return False, f"列集合不同：{sorted(set(a) ^ set(b))[:8]}"
    for col in a:
        x, y = a[col], b[col]
        if set(x) != set(y):
            return False, f"{col} 统计字段不同"
        for k in x:
            if not np.isclose(float(x[k]), float(y[k]), rtol=0, atol=atol):
                return False, f"{col}.{k}: {x[k]} != {y[k]}"
    return True, ""


def validate_dataset(data_dir):
    """导出目录的离线验收测试（不连数据库）。

    覆盖审查建议 12.1 / 12.2（结构性子集）/ 12.3 / 12.6：
      1. 文件齐全（训练导出含 train_preview.csv）、矩阵有限、形状与列
         定义一致、日期有序唯一；
      2. 样本索引与标签一致（target_pos == end_pos + 1、窗口长度、
         日期与位置一致、目标晚于输入）；train_preview.csv 与矩阵逐值
         一致；
      3. y_raw 可由相邻后复权收盘价重算，y_scaled 与矩阵末列一致；
      4. 切分边界：train/val/test 的目标日期互不越界；
      5. 对数收益率 expm1 与简单收益率一致；
      6. 审计 available <= decision（未来数据不可用断言的复核）；
      7. 验证/测试期数值扰动后重拟合，统计量与特征列不变；
      8. 未来行扰动后重 transform，历史行矩阵不变；
      9. latest.npz 形状与有限性。
    返回是否全部通过；任何失败项都会打印。
    """
    root = Path(data_dir)
    checks = []

    def record(name, ok, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if ok else "fail",
            "detail": str(detail),
        })
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {name}"
        if detail:
            line += f"：{detail}"
        print(line)
        return ok

    # 必需文件由 manifest 登记的模式决定，而不是"缺 samples.csv 就当
    # infer 跳过关键检查"——训练导出意外丢失样本文件必须判为验收失败。
    required = [
        "feature_matrix.npy", "dates.npy", "preprocessing.json",
        "aligned_values.csv", "alignment_audit.csv", "manifest.json",
        "features_unscaled.csv",
    ]
    missing = [f for f in required if not (root / f).exists()]
    record("required_files", not missing, f"缺失 {missing}" if missing else "")
    if missing:
        return False

    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    export_mode = manifest.get("mode")
    if export_mode not in ("prepare", "infer"):
        record(
            "manifest_mode", False,
            f"manifest.mode={export_mode!r} 无法识别，无法判定必需文件",
        )
        return False

    has_samples = (root / "samples.csv").exists()
    if export_mode == "prepare" and not has_samples:
        record(
            "samples_required_for_prepare", False,
            "训练导出（prepare）缺失 samples.csv，不能按 infer 跳过检查",
        )
        return False
    if export_mode == "prepare":
        required.append("train_preview.csv")
        missing = [f for f in required if not (root / f).exists()]
        record(
            "required_files_prepare", not missing,
            f"缺失 {missing}" if missing else "",
        )
        if missing:
            return False

    matrix = np.load(root / "feature_matrix.npy")
    dates = pd.DatetimeIndex(np.load(root / "dates.npy"))
    prep = json.loads(
        (root / "preprocessing.json").read_text(encoding="utf-8")
    )
    raw = pd.read_csv(
        root / "aligned_values.csv", index_col=0, parse_dates=True
    )
    columns = prep["feature_columns"]
    return_type = prep["config"].get("return_type", "simple")
    seq_len = int(prep["config"]["seq_len"])

    # infer 导出不生成 samples.csv / 训练统计复核，样本类检查跳过。
    samples = (
        pd.read_csv(root / "samples.csv") if has_samples else None
    )
    if samples is None:
        record(
            "sample_checks_skipped", True,
            "无 samples.csv（infer 导出），跳过样本与切分检查",
        )
    elif "label_valid" not in samples.columns:
        record(
            "label_valid_column", False,
            "samples.csv 缺少 label_valid 列（旧版导出），请用当前版本"
            "重新导出后再验收",
        )
        samples = None

    record("matrix_finite", bool(np.isfinite(matrix).all()))
    record(
        "matrix_shape",
        matrix.ndim == 2
        and matrix.shape[0] == len(dates)
        and matrix.shape[1] == len(columns),
        f"{matrix.shape} vs ({len(dates)}, {len(columns)})",
    )
    record(
        "dates_sorted_unique",
        bool(dates.is_monotonic_increasing and dates.is_unique),
    )
    record("target_last_column", columns[-1] == TARGET)

    train_end = pd.Timestamp(prep["train_end"])
    split_layout = prep.get("split_layout", "train_val_test")
    val_end = (
        pd.Timestamp(prep["val_end"])
        if prep.get("val_end") is not None else None
    )
    if split_layout == "train_test" and val_end is not None:
        raise ValueError("train_test 布局的 preprocessing 不应包含 val_end")

    if samples is not None:
        sp = samples["start_pos"].to_numpy(dtype=int)
        ep = samples["end_pos"].to_numpy(dtype=int)
        tp = samples["target_pos"].to_numpy(dtype=int)
        # 先验证位置边界再索引，坏数据输出验收失败项而不是 IndexError。
        in_range = bool(
            (sp >= 0).all()
            and (ep < len(dates)).all()
            and (tp < len(dates)).all()
        )
        record(
            "sample_positions_in_range", in_range,
            f"max_pos={max(tp.max(), ep.max()) if len(tp) else -1}, "
            f"dates={len(dates)}",
        )

        allowed_splits = (
            {"train", "test"} if split_layout == "train_test"
            else {"train", "val", "test"}
        )
        record(
            "split_values_allowed",
            set(samples["split"].dropna().unique()) <= allowed_splits,
            f"observed={sorted(samples['split'].dropna().unique())}",
        )

        label_valid = samples["label_valid"].to_numpy(dtype=bool)
        record(
            "label_invalid_rows_nan_y_raw",
            bool(samples.loc[~label_valid, "y_raw"].isna().all()),
            f"unscoreable={int((~label_valid).sum())}",
        )

        if in_range:
            record(
                "target_pos_follows_window", bool((tp == ep + 1).all())
            )
            record(
                "window_length", bool((ep - sp + 1 == seq_len).all()),
                f"seq_len={seq_len}",
            )
            input_dates = pd.to_datetime(samples["input_date"]).to_numpy()
            target_dates = pd.to_datetime(samples["target_date"]).to_numpy()
            record(
                "sample_pos_date_consistency",
                bool(
                    (dates[ep] == input_dates).all()
                    and (dates[tp] == target_dates).all()
                ),
            )
            record(
                "target_after_input", bool((dates[tp] > dates[ep]).all())
            )

            close = raw["px_close"].to_numpy(dtype=float)
            if return_type == "log":
                y_ref = np.log(close[tp] / close[tp - 1])
            else:
                y_ref = close[tp] / close[tp - 1] - 1.0
            y_raw = samples["y_raw"].to_numpy(dtype=float)
            err = (
                float(np.max(np.abs(y_ref[label_valid] - y_raw[label_valid])))
                if label_valid.any() else 0.0
            )
            record(
                "y_raw_recompute_from_close", err <= 1e-9,
                f"max|Δ|={err:.2e}",
            )

            p = prep["statistics"][TARGET]
            y_scaled_ref = (y_raw - p["mean"]) / p["scale"]
            err2 = (
                float(np.max(np.abs(
                    matrix[tp[label_valid], -1] - y_scaled_ref[label_valid]
                )))
                if label_valid.any() else 0.0
            )
            record(
                "y_scaled_matches_matrix", err2 <= 1e-4,
                f"max|Δ|={err2:.2e}",
            )

            if return_type == "log":
                simple_ref = close[tp] / close[tp - 1] - 1.0
                err3 = (
                    float(np.max(np.abs(
                        np.expm1(y_raw[label_valid])
                        - simple_ref[label_valid]
                    )))
                    if label_valid.any() else 0.0
                )
                record(
                    "log_to_simple_consistency", err3 <= 1e-12,
                    f"max|Δ|={err3:.2e}",
                )

            tdate = pd.to_datetime(samples["target_date"])
            split = samples["split"]
            if split_layout == "train_test":
                ok = (
                    (tdate[split == "train"] <= train_end).all()
                    and (tdate[split == "test"] > train_end).all()
                    and (split == "val").sum() == 0
                )
            else:
                ok = (
                    (tdate[split == "train"] <= train_end).all()
                    and (tdate[split == "val"] > train_end).all()
                    and (tdate[split == "val"] <= val_end).all()
                    and (tdate[split == "test"] > val_end).all()
                )
            record("split_boundaries", bool(ok))

            # rolling 导出（--start）时训练目标日不得早于登记的训练起点。
            start_value = (prep.get("sample_selection") or {}).get("start")
            if start_value is not None:
                record(
                    "train_start_lower_bound",
                    bool(
                        (tdate[split == "train"] >= pd.Timestamp(start_value)).all()
                    ),
                    f"start={start_value}",
                )

        # 训练前数据预览 CSV 与矩阵逐值一致（导出即训练输入的复核）。
        # CSV 十进制往返存在浮点表示噪声（float32 最短表示 ~1e-6 相对
        # 误差），用相对容差比较；真实数据错误远大于该量级。
        preview = pd.read_csv(
            root / "train_preview.csv", index_col=0, parse_dates=True
        )
        err = float(
            np.nanmax(np.abs(
                preview[columns].to_numpy(dtype=float)
                - matrix.astype(np.float64)
            )) if len(preview) == len(dates) else np.inf
        )
        preview_ok = (
            list(preview.columns) == columns + ["y_raw", "split"]
            and len(preview) == len(dates)
            and np.allclose(
                preview[columns].to_numpy(dtype=float),
                matrix.astype(np.float64), rtol=1e-6, atol=1e-9,
            )
        )
        record(
            "train_preview_matches_matrix", bool(preview_ok),
            f"shape={preview.shape} vs {matrix.shape}, max|Δ|={err:.2e}",
        )

    audit = pd.read_csv(root / "alignment_audit.csv", index_col=0)
    decision = pd.to_datetime(audit["decision_time"])
    bad = 0
    for col in audit.columns:
        if col.endswith("__available"):
            available = pd.to_datetime(audit[col])
            bad += int((available.notna() & (available > decision)).sum())
    record("audit_available_le_decision", bad == 0, f"违规 {bad} 行")

    features = pd.read_csv(
        root / "features_unscaled.csv", index_col=0, parse_dates=True
    )
    perturbed = features.copy()
    eval_mask = perturbed.index > train_end
    perturbed.loc[eval_mask, columns] = (
        perturbed.loc[eval_mask, columns] * 1.5 + 0.25
    )
    if samples is not None:
        groups = {}
        for col in columns:
            if col == TARGET:
                continue  # TARGET 由 fit_preprocessing 固定追加在末尾
            base = re.sub(r"__(missing|age_days)$", "", col)
            groups.setdefault(base, []).append(col)
        try:
            refit = fit_preprocessing(
                perturbed, samples, groups,
                float(prep.get("min_feature_coverage", 0.60)),
                int(prep.get("min_observations", 100)),
            )
            same_columns = refit["feature_columns"] == columns
            same_stats, why = _stats_close(
                refit["statistics"], prep["statistics"]
            )
            record(
                "preprocessing_immune_to_eval_data",
                bool(same_columns and same_stats),
                "" if same_columns and same_stats else (why or "特征列不同"),
            )
        except Exception as exc:
            record("preprocessing_immune_to_eval_data", False, str(exc))

    original = transform(features, prep)
    after = transform(perturbed, prep)
    history_mask = np.asarray(features.index <= train_end)
    record(
        "future_perturbation_invariance",
        bool(np.array_equal(original[history_mask], after[history_mask])),
    )

    gaps = dates.to_series().diff().dt.days.dropna()
    record(
        "calendar_gap_info", True,
        f"最大相邻交易日间隔 {int(gaps.max())} 天（软信息，不判失败）",
    )

    if (root / "latest.npz").exists():
        latest = np.load(root / "latest.npz")
        ok = (
            latest["X"].shape == (1, seq_len, len(columns))
            and bool(np.isfinite(latest["X"]).all())
        )
        record(
            "latest_window", bool(ok),
            f"shape={latest['X'].shape}, input_date={latest['input_date']}",
        )

    passed = sum(1 for c in checks if c["status"] == "pass")
    all_ok = passed == len(checks)
    print(f"validate {root}: {passed}/{len(checks)} 项通过" + (
        "" if all_ok else "，存在失败项"
    ))
    return all_ok


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
