from __future__ import annotations

import pandas as pd

from src.backtest import entry_event_mask, simulate, simulate_with_trace


def test_new_signals_open_independent_overlapping_trades() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.0, 100.0],
            "high": [100.5, 101.5, 102.5, 101.5, 100.5],
            "low": [99.5, 100.5, 101.5, 100.5, 99.5],
            "ret": [0.0, 0.01, 0.00990099, -0.00980392, -0.00990099],
        }
    )
    signal = pd.Series([1, 0, 0, -1, 0], dtype="int8")

    result = simulate(
        df,
        signal=signal,
        cost_bps=0.0,
        trade_size_pct=1.0,
        long_only=False,
        take_profit_pct=None,
        stop_loss_pct=None,
    )

    assert result.position.tolist() == [0.0, 1.0, 1.0, 1.0, 2.0]


def test_long_only_ignores_persistent_buy_state_until_new_entry_event() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "high": [100.0, 101.5, 102.5, 103.5],
            "low": [100.0, 100.5, 101.5, 102.5],
            "ret": [0.0, 0.01, 0.00990099, 0.00980392],
        }
    )
    signal = pd.Series([1, 1, -1, 0], dtype="int8")

    result = simulate(
        df,
        signal=signal,
        cost_bps=0.0,
        trade_size_pct=1.0,
        long_only=True,
        take_profit_pct=None,
        stop_loss_pct=None,
    )

    assert result.position.iloc[1] == 1.0
    assert result.position.iloc[2] == 1.0
    assert result.position.iloc[3] == 1.0


def test_short_trade_hits_take_profit_on_intrabar_low() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 99.0, 99.0],
            "high": [100.0, 100.5, 99.5],
            "low": [100.0, 93.5, 98.5],
            "ret": [0.0, -0.01, 0.0],
        }
    )
    signal = pd.Series([-1, 0, 0], dtype="int8")

    result = simulate(
        df,
        signal=signal,
        cost_bps=0.0,
        trade_size_pct=1.0,
        long_only=False,
        take_profit_pct=0.06,
        stop_loss_pct=0.03,
    )

    assert abs(float(result.net_returns.iloc[1]) - 0.06) < 1e-12
    assert result.position.tolist() == [0.0, 0.0, 0.0]


def test_entry_event_mask_marks_each_executable_signal() -> None:
    signal = pd.Series([1, 1, 0, -1, -1, 1], dtype="int8")

    mask = entry_event_mask(signal=signal, long_only=False)

    assert list(mask[mask].index) == [0, 3, 5]


def test_simulate_with_trace_supports_multiple_open_trades() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "ret": [0.0, 0.01, 0.00990099, 0.00980392],
        }
    )
    trace = simulate_with_trace(
        df,
        signal=pd.Series([1, 1, -1, 0], dtype="int8"),
        long_only=True,
        take_profit_pct=None,
        stop_loss_pct=None,
    )

    assert int(trace.events.loc[0, "entry_count"]) == 1
    assert int(trace.events.loc[1, "entry_count"]) == 0
    assert int(trace.events.loc[2, "entry_count"]) == 0
    assert len(trace.trade_log) == 1
    assert (trace.trade_log["status"] == "open").all()


def test_simulate_with_trace_opens_new_trade_only_when_signal_changes() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "ret": [0.0, 0.01, 0.00990099, 0.00980392, 0.00970874],
        }
    )
    trace = simulate_with_trace(
        df,
        signal=pd.Series([1, 1, 0, 1, 1], dtype="int8"),
        long_only=True,
        take_profit_pct=None,
        stop_loss_pct=None,
    )

    assert int(trace.events["entry_count"].sum()) == 2
    assert len(trace.trade_log) == 2
    assert (trace.trade_log["status"] == "open").all()


def test_trade_size_pct_scales_pnl_to_fraction_of_equity() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 106.0, 106.0],
            "high": [100.0, 106.5, 106.5],
            "low": [100.0, 99.5, 105.5],
            "ret": [0.0, 0.06, 0.0],
        }
    )

    trace = simulate_with_trace(
        df,
        signal=pd.Series([1, 0, 0], dtype="int8"),
        cost_bps=0.0,
        initial_cash=1.0,
        trade_size_pct=0.01,
        long_only=True,
        take_profit_pct=0.06,
        stop_loss_pct=0.03,
    )

    assert abs(float(trace.result.equity.iloc[-1]) - 1.0006) < 1e-12


def test_simulate_with_trace_records_short_trade() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 99.0, 98.0],
            "high": [100.0, 100.5, 98.5],
            "low": [100.0, 93.5, 97.5],
            "ret": [0.0, -0.01, -0.01010101],
        }
    )
    trace = simulate_with_trace(
        df,
        signal=pd.Series([-1, 0, 0], dtype="int8"),
        long_only=False,
        take_profit_pct=0.06,
        stop_loss_pct=0.03,
    )

    assert bool(trace.events.loc[0, "entry_event"]) is True
    assert trace.events.loc[0, "entry_direction"] == -1
    assert bool(trace.events.loc[1, "exit_event"]) is True
    assert trace.events.loc[1, "exit_reason"] == "take_profit"
    assert trace.trade_log.loc[0, "side"] == "short"
