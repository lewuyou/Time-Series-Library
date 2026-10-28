# -*- coding: utf-8 -*-
"""筛选符合训练要求的股票池（完整流程第 1 步：只读数据库，输出股票清单 CSV + 筛选元信息）。

用法：
    python select_universe.py --db ../AutoTS/download/autots.duckdb \
        --out dataset/global_001/universe.csv \
        --top-mv 3000 --min-years 3 --baidu-fresh-days 30

入选要求（对应需求）：
    1. 市值排在前 --top-mv 位（默认 3000，按 stock_list.total_mv 降序）；
    2. 辅助数据里有百度数据：baidu 表中存在 keyword = 当前股票简称 的记录，
       且仍在更新（最新观测日距库内最新行情日不超过 --baidu-fresh-days）；
    3. 辅助数据里有两融数据：rzrq 表中存在该股票记录；
    4. 总数据量不小于 --min-years 年（默认 3 年）：后复权行情
       跨度 >= 365*min-years 天 且 有效交易日数（count(DISTINCT date)）
       >= 244*min-years 天。

说明：
    - 本脚本只是入库数据的粗筛（快速排除明显不合格的股票）；百度/两融
      特征是否真的进入模型，由 export_global.py 在导出后按
      preprocessing.json 的 feature_columns 复检（训练期覆盖率不足 0.6
      的列会被 dataset.py 丢弃，复检不通过的股票不训练）。
    - 这里的"三年"只是行情口径，不代表百度/两融各自都有三年数据，
      后续仍须根据训练期覆盖率复检。
    - 百度关键词按 dataset.py 的口径取"当前股票简称"。
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from dataset import json_write, sql_symbol

# A 股一年约 244 个交易日；三年约 732。
TRADING_DAYS_PER_YEAR = 244


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="../AutoTS/download/autots.duckdb")
    p.add_argument("--out", default="dataset/universe_run_001/universe.csv")
    p.add_argument("--top-mv", type=int, default=3000, help="市值排名上限")
    p.add_argument("--min-years", type=float, default=3.0, help="最少历史年限")
    p.add_argument(
        "--baidu-fresh-days", type=int, default=30,
        help="百度关键词最新观测日距库内最新行情日的最大天数",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_mv < 1 or args.min_years <= 0:
        raise ValueError("--top-mv 必须 >= 1，--min-years 必须 > 0")

    con = duckdb.connect(args.db, read_only=True)
    try:
        asof = con.execute(
            "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
        ).fetchone()[0]
        if asof is None:
            raise ValueError("数据库没有后复权行情")
        asof = pd.Timestamp(asof).normalize()

        stocks = con.execute(
            f"""
            SELECT {sql_symbol("symbol")} AS symbol, name, exchange,
                   industry_l2, total_mv,
                   row_number() OVER (
                       ORDER BY total_mv DESC NULLS LAST
                   ) AS mv_rank
            FROM stock_list
            QUALIFY mv_rank <= ?
            ORDER BY mv_rank
            """,
            [args.top_mv],
        ).fetchdf()

        quotes = con.execute(
            f"""
            SELECT {sql_symbol("symbol")} AS symbol,
                   count(DISTINCT date) AS quote_days,
                   min(date) AS first_quote, max(date) AS last_quote
            FROM akshare_tx
            WHERE adjust = 'hfq'
            GROUP BY 1
            """
        ).fetchdf()
        quotes["first_quote"] = pd.to_datetime(quotes["first_quote"])
        quotes["last_quote"] = pd.to_datetime(quotes["last_quote"])
        quotes["span_days"] = (
            quotes["last_quote"] - quotes["first_quote"]
        ).dt.days

        rzrq = con.execute(
            f"""
            SELECT {sql_symbol("code")} AS symbol,
                   count(*) AS rzrq_rows, max(date) AS rzrq_last
            FROM rzrq
            GROUP BY 1
            """
        ).fetchdf()

        baidu = con.execute(
            """
            SELECT keyword, count(DISTINCT date) AS baidu_days,
                   max(date) AS baidu_last
            FROM baidu
            GROUP BY 1
            """
        ).fetchdf()
    finally:
        con.close()

    baidu["baidu_last"] = pd.to_datetime(baidu["baidu_last"])
    rzrq["rzrq_last"] = pd.to_datetime(rzrq["rzrq_last"])

    min_span = int(365 * args.min_years)
    min_days = int(TRADING_DAYS_PER_YEAR * args.min_years)
    fresh_limit = asof - pd.Timedelta(days=args.baidu_fresh_days)

    df = stocks.merge(quotes, on="symbol", how="left")
    df = df.merge(rzrq, on="symbol", how="left")
    df = df.merge(
        baidu, left_on="name", right_on="keyword", how="left"
    )

    stages = {"top_mv": len(df)}
    df = df[df["span_days"].notna()]
    stages["has_quotes"] = len(df)

    df = df[df["span_days"] >= min_span]
    stages[f"span_ge_{min_span}d"] = len(df)
    df = df[df["quote_days"] >= min_days]
    stages[f"quote_days_ge_{min_days}"] = len(df)

    df = df[df["rzrq_rows"].notna() & (df["rzrq_rows"] > 0)]
    stages["has_rzrq"] = len(df)

    df = df[
        df["baidu_days"].notna()
        & (df["baidu_days"] > 0)
        & (df["baidu_last"] >= fresh_limit)
    ]
    stages["has_fresh_baidu"] = len(df)

    df = df.sort_values("mv_rank").reset_index(drop=True)
    out_cols = [
        "symbol", "name", "exchange", "industry_l2", "total_mv", "mv_rank",
        "quote_days", "first_quote", "last_quote", "span_days",
        "baidu_days", "baidu_last", "rzrq_rows", "rzrq_last",
    ]
    df = df[out_cols]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    meta = {
        "db": str(Path(args.db).resolve()),
        "as_of": str(asof.date()),
        "criteria": {
            "top_mv": args.top_mv,
            "min_years": args.min_years,
            "min_span_days": min_span,
            "min_quote_days": min_days,
            "baidu_fresh_days": args.baidu_fresh_days,
        },
        "stages": stages,
        "selected": len(df),
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    json_write(out.with_name("universe_meta.json"), meta)

    print(f"as_of={asof.date()} | 筛选漏斗：{json.dumps(stages, ensure_ascii=False)}")
    print(f"[OK] 入选 {len(df)} 只股票 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
