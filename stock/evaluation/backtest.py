"""可执行回测：与每日选股同一信号集合，成交假设显式。

`daily_hit` 只回答"预测命中率"，隐含按目标日收盘价成交；真实交易在
信号形成后的下一交易日开盘才可能成交。本模块并列两种口径：

- close_to_close：直接用标签 `y_true`（t 收盘 → t+1 收盘，后复权），
  与 `daily_hit` 完全同源，作为上界/理论口径；
- executable：t+1 开盘买入，持有 hold_days 个交易日，按收盘卖出，
  并扣除双边手续费与滑点，作为可交易口径。

信号必须复用 `stock.selection.rules.hybrid_signal`，且传入与
`daily_hit` 相同的 eligible 集合（当日有预测的全部股票），否则排名
分母不同会导致信号集合漂移。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock.evaluation.daily_hit import daily_hit
from stock.selection.rules import hybrid_signal
from stock.utils import write_json


def compute_signals(predictions, q_pct=5.0, floor_pct=0.5, eligible=None):
    """逐日跑混合规则，返回通过信号的记录（与 daily_hit 同口径）。

    命中分析中 y_pred 是小数收益率，floor 0.5% 对应 0.5/100；
    这里显式传 floor_decimal + pred_unit="decimal"，避免单位混用。
    """
    parts = []
    for date, part in predictions.groupby("target_date"):
        part = part.copy()
        same_day = part["symbol"].astype(str).tolist()
        allowed = None if eligible is None else {str(s) for s in eligible}
        elig = same_day if allowed is None else [
            s for s in same_day if s in allowed
        ]
        rule = hybrid_signal(
            part["y_pred"], q_pct,
            floor_decimal=floor_pct / 100.0, pred_unit="decimal",
            symbol=part["symbol"], eligible=elig,
        )
        part["rank"] = rule["rank"].to_numpy()
        part["quantile_pass"] = rule["quantile_pass"].to_numpy()
        part["floor_pass"] = rule["floor_pass"].to_numpy()
        part["signal"] = rule["signal"].to_numpy()
        parts.append(part)
    if not parts:
        return predictions.iloc[0:0].assign(
            rank=[], quantile_pass=[], floor_pass=[], signal=[]
        )
    return pd.concat(parts, ignore_index=True)


def load_prices(path):
    """读取 symbol,date,open,close 行情（日期升序）。"""
    frame = pd.read_csv(path, dtype={"symbol": str})
    missing = {"symbol", "date", "open", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"行情文件缺少列：{sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["open", "close"])
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _wide(prices, column):
    return prices.pivot(index="date", columns="symbol", values=column)


def executable_returns(signals, prices, hold_days=1, fee_bps=0.0,
                       slippage_bps=0.0):
    """逐条信号的持有期收益（t+1 开盘买，持有 hold_days 日后收盘卖）。

    成本：entry=open*(1+slip)，exit=close*(1-slip)；
    ret = exit/entry*(1-fee)^2 - 1（买入、卖出各计一次手续费）。
    """
    if hold_days < 1:
        raise ValueError("hold_days 必须 >= 1")
    open_wide = _wide(prices, "open")
    close_wide = _wide(prices, "close")
    dates = list(open_wide.index)
    position = {d: i for i, d in enumerate(dates)}
    slip = float(slippage_bps) / 10000.0
    fee = float(fee_bps) / 10000.0

    out = []
    for row in signals.itertuples(index=False):
        entry_date = str(row.target_date)
        index = position.get(entry_date)
        exit_index = None if index is None else index + int(hold_days) - 1
        if exit_index is None or exit_index >= len(dates) or not _has(
            open_wide, close_wide, entry_date, dates[exit_index], row.symbol
        ):
            out.append({
                "symbol": row.symbol, "target_date": entry_date,
                "entry_date": entry_date, "exit_date": None,
                "exec_return": np.nan,
            })
            continue
        exit_date = dates[exit_index]
        entry = open_wide.at[entry_date, row.symbol] * (1.0 + slip)
        exit_price = close_wide.at[exit_date, row.symbol] * (1.0 - slip)
        value = exit_price / entry * (1.0 - fee) ** 2 - 1.0
        out.append({
            "symbol": row.symbol, "target_date": entry_date,
            "entry_date": entry_date, "exit_date": exit_date,
            "exec_return": value,
        })
    return pd.DataFrame(out)


def _has(open_wide, close_wide, entry_date, exit_date, symbol):
    if symbol not in open_wide.columns or symbol not in close_wide.columns:
        return False
    return bool(
        np.isfinite(open_wide.at[entry_date, symbol])
        and np.isfinite(close_wide.at[exit_date, symbol])
        and open_wide.at[entry_date, symbol] > 0
        and close_wide.at[exit_date, symbol] > 0
    )


def _portfolio(daily_returns, all_dates):
    """等权组合（无信号日空仓），返回净值/回撤/逐日收益。"""
    series = pd.Series(daily_returns).reindex(all_dates).fillna(0.0)
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return series, equity, drawdown


def _equity_records(equity):
    frame = equity.rename("equity").rename_axis("target_date").reset_index()
    return _records(frame)


def _records(frame):
    """DataFrame → JSON 安全记录（NaN → None，numpy 标量 → Python）。"""
    data = frame.where(pd.notna(frame), None)
    return [
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
        }
        for row in data.to_dict(orient="records")
    ]


def _turnover(signal_sets, dates):
    rows = []
    previous = set()
    for date in dates:
        current = signal_sets.get(date, set())
        added = len(current - previous)
        removed = len(previous - current)
        rows.append({
            "target_date": date,
            "n_positions": len(current),
            "added": added,
            "removed": removed,
            "turnover": (
                0.5 * (added + removed) / max(len(current), len(previous), 1)
            ),
        })
        previous = current
    return pd.DataFrame(rows)


def run_backtest(predictions, prices=None, q_pct=5.0, floor_pct=0.5,
                 hold_days=1, fee_bps=0.0, slippage_bps=0.0,
                 eligible=None):
    """执行回测；prices=None 时只输出 close_to_close 口径。"""
    required = {"symbol", "target_date", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"预测记录缺少列：{sorted(missing)}")
    predictions = predictions.copy()
    predictions["symbol"] = predictions["symbol"].astype(str)
    predictions["target_date"] = predictions["target_date"].astype(str)

    scored = compute_signals(
        predictions, q_pct=q_pct, floor_pct=floor_pct, eligible=eligible
    )
    signals = scored[scored["signal"]].copy()
    dates = sorted(scored["target_date"].unique())
    signal_sets = {
        date: set(part["symbol"]) for date, part in signals.groupby("target_date")
    }
    turnover = _turnover(signal_sets, dates)

    report = {
        "assumptions": {
            "signal": f"每日 y_pred 前 {q_pct:g}%（ceil）且 y_pred > {floor_pct:g}%",
            "eligible": "当日有预测的全部股票（与 daily_hit 相同）",
            "close_to_close": "t 收盘 → t+1 收盘（后复权标签 y_true）",
            "executable": (
                f"t+1 开盘买入，持有 {hold_days} 个交易日后收盘卖出；"
                f"手续费 {fee_bps:g}bp/边，滑点 {slippage_bps:g}bp/边"
            ),
            "weights": "信号股等权，无信号日空仓",
            "turnover": "0.5*(新增+剔除)/max(期初,期末持仓数)，日频再平衡近似",
        },
        "days": len(dates),
        "signals": int(len(signals)),
        "signal_days": int(sum(1 for v in signal_sets.values() if v)),
        "close_to_close": None,
        "executable": None,
        "signals_detail": (
            _records(signals[[
                "symbol", "target_date", "y_pred", "y_true", "rank"
            ]]) if "y_true" in signals else []
        ),
    }

    if "y_true" in scored:
        closed = signals[["target_date", "symbol", "y_true"]].dropna()
        daily = closed.groupby("target_date")["y_true"].mean()
        series, equity, drawdown = _portfolio(daily, dates)
        hit = float((closed["y_true"] > 0).mean()) if len(closed) else None
        report["close_to_close"] = {
            "n": int(len(closed)),
            "hit_rate": hit,
            "daily_mean_return": float(daily.mean()) if len(daily) else None,
            "cumulative_return": float(equity.iloc[-1] - 1) if len(equity) else None,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else None,
            "equity": _equity_records(equity),
        }

    if prices is not None:
        trades = executable_returns(
            signals, prices, hold_days=hold_days, fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        priced = trades.dropna(subset=["exec_return"])
        daily = priced.groupby("target_date")["exec_return"].mean()
        series, equity, drawdown = _portfolio(daily, dates)
        report["executable"] = {
            "n": int(len(priced)),
            "unpriced": int(len(trades) - len(priced)),
            "hit_rate": (
                float((priced["exec_return"] > 0).mean()) if len(priced) else None
            ),
            "daily_mean_return": float(daily.mean()) if len(daily) else None,
            "cumulative_return": float(equity.iloc[-1] - 1) if len(equity) else None,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else None,
            "avg_turnover": float(turnover["turnover"].mean()),
            "equity": _equity_records(equity),
        }
        report["trades"] = _records(trades)

    report["turnover"] = _records(turnover)
    return report, scored, signals


def daily_hit_reference(predictions, q_pct=5.0, floor_pct=0.5):
    """调用 daily_hit 作为对照（验收：信号集合必须完全一致）。"""
    frame = predictions.rename(columns={"y_true": "y_true"})
    if "y_true" not in frame:
        frame = frame.assign(y_true=0.0)
    daily, stats = daily_hit(frame, q_pct, floor_pct)
    return daily, stats


def parse_args():
    p = argparse.ArgumentParser(description="信号回测（收盘→收盘 / 可执行）")
    p.add_argument("--predictions", required=True,
                   help="CSV：symbol,target_date,y_pred[,y_true]")
    p.add_argument("--prices", help="CSV：symbol,date,open,close（可执行口径必需）")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quantile-pct", type=float, default=5.0)
    p.add_argument("--floor-pct", type=float, default=0.5)
    p.add_argument("--hold-days", type=int, default=1)
    p.add_argument("--fee-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--eligible-symbols", nargs="*",
                   help="可选：限定 eligible 集合（默认当日全部预测股票）")
    return p.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions, dtype={"symbol": str})
    prices = load_prices(args.prices) if args.prices else None
    report, _, signals = run_backtest(
        predictions, prices=prices, q_pct=args.quantile_pct,
        floor_pct=args.floor_pct, hold_days=args.hold_days,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        eligible=args.eligible_symbols,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "backtest_report.json", report)
    signals.to_csv(out / "backtest_signals.csv", index=False)
    if report["turnover"]:
        pd.DataFrame(report["turnover"]).to_csv(
            out / "backtest_turnover.csv", index=False
        )
    if report.get("trades"):
        pd.DataFrame(report["trades"]).to_csv(
            out / "backtest_trades.csv", index=False
        )
    print(json.dumps({
        key: report[key] for key in
        ("days", "signals", "signal_days", "assumptions")
    }, ensure_ascii=False, indent=2))
    for name in ("close_to_close", "executable"):
        block = report[name]
        if block:
            print(f"\n[{name}] " + json.dumps({
                k: block[k] for k in
                ("n", "hit_rate", "daily_mean_return",
                 "cumulative_return", "max_drawdown")
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
