"""交易日历：加载、指纹与历史前缀哈希（自 dataset.py 原样迁出）。"""

import hashlib

import pandas as pd

from stock.data.common import assert_unique, parse_dates
from stock.utils import sha256_file

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


