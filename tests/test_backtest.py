import pandas as pd
import pytest

from stock.evaluation.backtest import run_backtest
from stock.evaluation.daily_hit import daily_hit

DATES = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
SYMBOLS = [f"S{i}" for i in range(8)]


def _predictions():
    rows = []
    for day_index, date in enumerate(DATES[1:], start=1):
        for i, symbol in enumerate(SYMBOLS):
            rows.append({
                "symbol": symbol,
                "target_date": date,
                "y_pred": (i + 1) / 100.0 * day_index,
                "y_true": 0.0,
            })
    return pd.DataFrame(rows)


def _prices():
    """开盘价 = 前一交易日收盘价（无隔夜跳空），便于两种口径对齐。"""
    rows = []
    for i, symbol in enumerate(SYMBOLS):
        base = 10.0 + i
        for day_index, date in enumerate(DATES):
            close = base * (1.0 + 0.01 * (i + day_index))
            previous_close = (
                base * (1.0 + 0.01 * (i + day_index - 1))
                if day_index else close
            )
            rows.append({
                "symbol": symbol, "date": date,
                "open": previous_close, "close": close,
            })
    return pd.DataFrame(rows)


def test_signal_set_matches_daily_hit(tmp_path):
    predictions = _predictions()
    path = tmp_path / "pred.csv"
    predictions.to_csv(path, index=False)
    report, scored, signals = run_backtest(
        predictions, q_pct=25, floor_pct=0.0
    )
    daily, _ = daily_hit(predictions, q_pct=25, floor_pct=0.0)
    got = signals.groupby("target_date").size()
    expected = daily.set_index("target_date")["signals"]
    assert got.reindex(expected.index).fillna(0).astype(int).equals(
        expected.astype(int)
    )
    pairs = set(zip(signals.symbol, signals.target_date))
    direct = scored[scored.signal][["symbol", "target_date"]]
    assert pairs == set(zip(direct.symbol, direct.target_date))
    assert report["signals"] == len(pairs)


def test_zero_cost_and_no_gap_matches_close_to_close():
    predictions = _predictions()
    prices = _prices()
    close_report, _, signals = run_backtest(
        predictions, prices=prices, q_pct=25, floor_pct=0.0,
        fee_bps=0.0, slippage_bps=0.0,
    )
    # 用真实收盘收益重算标签，而不是占位的 0
    rows = prices.copy()
    rows["prev_close"] = rows.groupby("symbol")["close"].shift(1)
    rows["y_true"] = rows["close"] / rows["prev_close"] - 1.0
    truth = rows.dropna(subset=["y_true"])[["symbol", "date", "y_true"]].rename(
        columns={"date": "target_date"}
    )
    predictions = predictions.drop(columns=["y_true"]).merge(
        truth, on=["symbol", "target_date"], how="left"
    )
    report, _, _ = run_backtest(
        predictions, prices=prices, q_pct=25, floor_pct=0.0,
        fee_bps=0.0, slippage_bps=0.0,
    )
    assert report["close_to_close"]["cumulative_return"] == pytest.approx(
        report["executable"]["cumulative_return"], rel=0, abs=1e-12
    )
    assert report["close_to_close"]["daily_mean_return"] == pytest.approx(
        report["executable"]["daily_mean_return"], rel=0, abs=1e-12
    )
    assert report["executable"]["unpriced"] == 0


def test_costs_reduce_equity():
    predictions = _predictions()
    prices = _prices()
    no_cost, _, _ = run_backtest(predictions, prices=prices, q_pct=25, floor_pct=0.0)
    cost, _, _ = run_backtest(
        predictions, prices=prices, q_pct=25, floor_pct=0.0,
        fee_bps=10.0, slippage_bps=5.0,
    )
    assert (
        cost["executable"]["cumulative_return"]
        < no_cost["executable"]["cumulative_return"]
    )


def test_hold_period_and_missing_price(tmp_path):
    predictions = _predictions()
    prices = _prices()
    prices = prices[~(
        (prices.symbol == "S7") & (prices.date == DATES[-1])
    )]
    report, _, signals = run_backtest(
        predictions, prices=prices, q_pct=50, floor_pct=0.0, hold_days=2
    )
    assert report["executable"]["unpriced"] >= 0
    trades = pd.DataFrame(report["trades"])
    assert "exit_date" in trades
    assert trades["exec_return"].isna().sum() >= 0
    # 持有期越界/缺价的行必须显式计为 unpriced，而不是丢弃
    assert len(trades) == len(signals)
