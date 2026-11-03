# -*- coding: utf-8 -*-
"""筛选符合训练要求的股票池（完整流程第 1 步：只读数据库，输出股票清单 CSV + 筛选元信息）。

自 select_universe.py 原样迁入；json 写入改用共享 stock.utils.write_json
（原子写），股票代码标准化仍复用根目录 dataset.sql_symbol。

用法：
    python -m stock.selection.universe --db ../AutoTS/download/autots.duckdb \
        --out dataset/global_001/universe.csv \
        --top-mv 3000 --min-years 3 --baidu-fresh-days 30

    walk-forward / 历史回测按当时时点粗筛（部分 point-in-time）：
    python -m stock.selection.universe --db ... --out ... --as-of 2024-01-02

入选要求（对应需求）：
    1. 市值排在前 --top-mv 位（默认 3000，按 stock_list.total_mv 降序）；
    2. 辅助数据里有百度数据：baidu 表中存在 keyword = 当前股票简称 的记录，
       且仍在更新（最新观测日距 as-of 日不超过 --baidu-fresh-days）；
    3. 辅助数据里有两融数据：rzrq 表中存在该股票记录；
    4. 总数据量不小于 --min-years 年（默认 3 年）：后复权行情
       跨度 >= 365*min-years 天 且 有行情记录的不同日期数量
       （count(DISTINCT date)）>= 244*min-years 天。
       SQL 未逐行验证 OHLC 是否有效，真正的行情有效性由 dataset.py 判断。
    5. （提供 --as-of 时）截至 as-of 已上市（stock_list.list_date <= as-of）。

point-in-time（PIT）口径说明：
    - 库内没有历史市值、历史简称、历史上市/退市快照，本模块的 PIT 能力
      是部分实现，必须区分两类资格条件：
      * 可按时点评估（提供 --as-of 时按 as-of 日重算）：
        上市日期（list_date <= as-of）、行情历史/跨度/交易日数
        （date <= as-of）、两融是否存在当时记录（date <= as-of）、
        百度数据截至 as-of 是否仍新鲜（max(date) <= as-of 的部分）；
      * 仍是当前快照（无法 PIT，明确保留限制）：
        市值排名 top_mv（total_mv 是 stock_list 当前快照）、
        百度关键词 = 当前简称（name 是当前快照）、industry_l2 当前分类；
        且 stock_list 本身是当前在市股票清单，历史已退市股票天然缺失
        （快照层面的幸存者偏差）。
    - 因此 --as-of 产物是"部分 PIT"资格查询：能排除当时尚未上市的股票、
      排除当时行情/两融/百度已不满足条件的股票，但市值排名与简称映射
      仍是今天的口径；历史市值补齐前不应视为真正 PIT。
    - 提供显式 --as-of 时，市值排名在"截至 as-of 已上市"的股票范围内
      计算（先过滤再排名），未上市股票不占据名次槽位；缺省（最新时点）
      不过滤上市日期。

说明：
    - 本模块只是入库数据的粗筛（快速排除明显不合格的股票）；百度/两融
      特征是否真的进入模型，由 stock.data.export 在导出后按
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

from dataset import sql_symbol
from stock.utils import write_json as json_write

# A 股一年约 244 个交易日；三年约 732。
TRADING_DAYS_PER_YEAR = 244


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="../AutoTS/download/autots.duckdb")
    p.add_argument(
        "--out", default="dataset/global_001/universe.csv"
    )
    p.add_argument("--top-mv", type=int, default=3000, help="市值排名上限")
    p.add_argument("--min-years", type=float, default=3.0, help="最少历史年限")
    p.add_argument(
        "--baidu-fresh-days", type=int, default=30,
        help="百度关键词最新观测日距 as-of 日的最大天数",
    )
    p.add_argument(
        "--as-of", default=None,
        help="按历史时点粗筛的日期（YYYY-MM-DD）：上市日期、行情/两融/"
             "百度历史均按 <= as-of 重算；市值排名与简称仍是当前快照"
             "（部分 PIT，见模块说明）。缺省 = 库内最新行情日（当前行为）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_mv < 1 or args.min_years <= 0:
        raise ValueError("--top-mv 必须 >= 1，--min-years 必须 > 0")
    if args.baidu_fresh_days < 0:
        raise ValueError("--baidu-fresh-days 必须 >= 0")

    explicit_asof = None
    if args.as_of:
        explicit_asof = pd.Timestamp(args.as_of).normalize()

    con = duckdb.connect(args.db, read_only=True)
    try:
        latest = con.execute(
            "SELECT max(date) FROM akshare_tx WHERE adjust='hfq'"
        ).fetchone()[0]
        if latest is None:
            raise ValueError("数据库没有后复权行情")
        latest = pd.Timestamp(latest).normalize()

        if explicit_asof is None:
            asof = latest
        else:
            if explicit_asof > latest:
                raise ValueError(
                    f"--as-of {explicit_asof.date()} 晚于库内最新行情日 "
                    f"{latest.date()}"
                )
            asof = explicit_asof

        # 市值排名仍是 stock_list 当前快照（无历史市值表，无法 PIT）；
        # --as-of 时先过滤"截至 as-of 已上市"再排名：未上市股票不应
        # 占据 top-mv 名次槽位（否则 2025 年上市的新股会挤掉 2023 年
        # 排名内的股票）；排名加 symbol 决胜保证确定性。
        if explicit_asof is not None:
            stocks = con.execute(
                f"""
                SELECT {sql_symbol("symbol")} AS symbol, name, exchange,
                       industry_l2, total_mv, list_date,
                       row_number() OVER (
                           ORDER BY total_mv DESC NULLS LAST, symbol
                       ) AS mv_rank
                FROM stock_list
                WHERE list_date IS NOT NULL AND list_date <= ?
                QUALIFY mv_rank <= ?
                ORDER BY mv_rank
                """,
                [asof, args.top_mv],
            ).fetchdf()
        else:
            stocks = con.execute(
                f"""
                SELECT {sql_symbol("symbol")} AS symbol, name, exchange,
                       industry_l2, total_mv, list_date,
                       row_number() OVER (
                           ORDER BY total_mv DESC NULLS LAST, symbol
                       ) AS mv_rank
                FROM stock_list
                QUALIFY mv_rank <= ?
                ORDER BY mv_rank
                """,
                [args.top_mv],
            ).fetchdf()
        if stocks["symbol"].duplicated().any():
            raise ValueError("stock_list 标准化股票代码后存在重复")
        stocks["list_date"] = pd.to_datetime(stocks["list_date"])

        # 行情/两融/百度历史均可按时点重算：date <= as-of。
        quotes = con.execute(
            f"""
            SELECT {sql_symbol("symbol")} AS symbol,
                   count(DISTINCT date) AS quote_days,
                   min(date) AS first_quote, max(date) AS last_quote
            FROM akshare_tx
            WHERE adjust = 'hfq' AND date <= ?
            GROUP BY 1
            """,
            [asof],
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
            WHERE date <= ?
            GROUP BY 1
            """,
            [asof],
        ).fetchdf()

        baidu = con.execute(
            """
            SELECT keyword, count(DISTINCT date) AS baidu_days,
                   max(date) AS baidu_last
            FROM baidu
            WHERE source IN ('search_all', 'feed')
              AND value IS NOT NULL
              AND date <= ?
            GROUP BY 1
            """,
            [asof],
        ).fetchdf()
    finally:
        con.close()

    baidu["baidu_last"] = pd.to_datetime(baidu["baidu_last"])
    rzrq["rzrq_last"] = pd.to_datetime(rzrq["rzrq_last"])

    min_span = int(365 * args.min_years)
    min_days = int(TRADING_DAYS_PER_YEAR * args.min_years)
    fresh_limit = asof - pd.Timedelta(days=args.baidu_fresh_days)

    df = stocks.merge(
        quotes, on="symbol", how="left", validate="one_to_one"
    )
    df = df.merge(
        rzrq, on="symbol", how="left", validate="one_to_one"
    )
    df = df.merge(
        baidu,
        left_on="name",
        right_on="keyword",
        how="left",
        validate="many_to_one",
    )

    stages = {"top_mv": len(df)}
    if explicit_asof is not None:
        # PIT 资格纵深防御：SQL 排名前已过滤"截至 as-of 已上市"，
        # 这里再校验一次（防御 SQL 重构回归）。
        df = df[df["list_date"].notna() & (df["list_date"] <= asof)]
        stages["listed_by_as_of"] = len(df)
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
        "list_date", "quote_days", "first_quote", "last_quote", "span_days",
        "baidu_days", "baidu_last", "rzrq_rows", "rzrq_last",
    ]
    df = df[out_cols]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    meta = {
        "db": str(Path(args.db).resolve()),
        "as_of": str(asof.date()),
        "as_of_explicit": explicit_asof is not None,
        "criteria": {
            "top_mv": args.top_mv,
            "min_years": args.min_years,
            "min_span_days": min_span,
            "min_quote_days": min_days,
            "baidu_fresh_days": args.baidu_fresh_days,
        },
        # PIT 能力分类：明确区分可按时点评估与仍是当前快照的条件，
        # 避免"部分 PIT"被误当成真正 PIT 使用。
        "pit_status": {
            "pit_criteria": [
                "listed_by_as_of" if explicit_asof is not None else None,
                "quote_history_span_days",
                "quote_days",
                "has_rzrq",
                "has_fresh_baidu",
            ],
            "snapshot_criteria": [
                "top_mv_rank",
                "baidu_keyword_name_mapping",
                "industry_l2",
                "stock_list_membership",
            ],
            "limitations": [
                "库内没有历史市值表：total_mv 排名是 stock_list 当前快照",
                "库内没有历史简称表：百度关键词按当前简称匹配",
                "stock_list 是当前在市清单：历史已退市股票天然缺失"
                "（快照层面幸存者偏差）",
                "无辅助数据发布时间（published_at/first_seen_at），"
                "行情/两融/百度按观测日 <= as-of 近似 PIT",
            ],
        },
        "stages": stages,
        "selected": len(df),
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    if explicit_asof is None:
        meta["pit_status"]["pit_criteria"] = [
            c for c in meta["pit_status"]["pit_criteria"]
            if c is not None
        ]
    # meta 文件名跟随 CSV stem：缺省 universe.csv -> universe_meta.json
    # （与历史一致）；walk-forward 每折多套股票池（universe_selection/
    # universe_refit）的 meta 互相不覆盖。
    json_write(out.with_name(f"{out.stem}_meta.json"), meta)

    print(f"as_of={asof.date()} | 筛选漏斗：{json.dumps(stages, ensure_ascii=False)}")
    print(f"[OK] 入选 {len(df)} 只股票 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
