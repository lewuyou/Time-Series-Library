"""原始数据读取与特征序列规格（行情/百度/估值/两融/宏观）。"""

import hashlib
import re

import pandas as pd

from stock.data.common import (
    BAIDU_FIELDS,
    PRICE_FIELDS,
    RZRQ_FIELDS,
    VAL_FIELDS,
    as_numeric,
    assert_unique,
    norm_symbol,
    parse_dates,
    sql_symbol,
)

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


