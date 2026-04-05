from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.config import BacktestConfig, ExperimentConfig
from src.run.artifacts import collect_trade_rows, trace_to_metrics_row


def run_trace(
    simulate_with_trace_fn: Callable[..., Any],
    market: pd.DataFrame,
    signal: pd.Series,
    backtest_cfg: BacktestConfig,
) -> Any:
    return simulate_with_trace_fn(
        market,
        signal=signal,
        cost_bps=backtest_cfg.cost_bps,
        initial_cash=backtest_cfg.initial_cash,
        trade_size_pct=backtest_cfg.trade_size_pct,
        long_only=backtest_cfg.long_only,
        take_profit_pct=backtest_cfg.take_profit_pct,
        stop_loss_pct=backtest_cfg.stop_loss_pct,
    )


def build_buy_and_hold_signal(index: pd.Index) -> pd.Series:
    signal = pd.Series(0, index=index, dtype="int8")
    if len(signal) > 0:
        signal.iloc[0] = 1
    return signal


def run_buy_and_hold_trace(
    simulate_with_trace_fn: Callable[..., Any],
    market: pd.DataFrame,
    initial_cash: float,
) -> Any:
    return simulate_with_trace_fn(
        market,
        signal=build_buy_and_hold_signal(market.index),
        cost_bps=0.0,
        initial_cash=initial_cash,
        trade_size_pct=1.0,
        long_only=True,
        take_profit_pct=None,
        stop_loss_pct=None,
    )


def run_full_sample_suite(
    market: pd.DataFrame,
    baseline_signals: dict[str, pd.Series],
    cfg: ExperimentConfig,
    annualization: int,
    simulate_with_trace_fn: Callable[..., Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Any], Any]:
    metrics_rows: list[dict[str, Any]] = []
    trades_frames: list[pd.DataFrame] = []
    full_traces: dict[str, Any] = {}

    for strategy_name, strategy_signal in baseline_signals.items():
        trace = run_trace(simulate_with_trace_fn, market, strategy_signal, cfg.backtest)
        full_traces[strategy_name] = trace
        metrics_rows.append(trace_to_metrics_row(strategy_name, "full", trace, annualization=annualization))
        trades_frames.append(collect_trade_rows(strategy_name, "full", trace.trade_log))

    buy_hold_trace = run_buy_and_hold_trace(simulate_with_trace_fn, market, cfg.backtest.initial_cash)
    metrics_rows.append(trace_to_metrics_row("buy_and_hold", "full", buy_hold_trace, annualization=annualization))
    trades_frames.append(collect_trade_rows("buy_and_hold", "full", buy_hold_trace.trade_log))
    return metrics_rows, trades_frames, full_traces, buy_hold_trace


def run_oos_baselines(
    oos_market: pd.DataFrame,
    oos_index: pd.Index,
    baseline_signals: dict[str, pd.Series],
    cfg: ExperimentConfig,
    annualization: int,
    simulate_with_trace_fn: Callable[..., Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], Any]:
    metrics_rows: list[dict[str, Any]] = []
    trades_frames: list[pd.DataFrame] = []

    for strategy_name, strategy_signal in baseline_signals.items():
        trace = run_trace(simulate_with_trace_fn, oos_market, strategy_signal.loc[oos_index], cfg.backtest)
        metrics_rows.append(trace_to_metrics_row(strategy_name, "oos", trace, annualization=annualization))
        trades_frames.append(collect_trade_rows(strategy_name, "oos", trace.trade_log))

    buy_hold_oos_trace = run_buy_and_hold_trace(simulate_with_trace_fn, oos_market, cfg.backtest.initial_cash)
    metrics_rows.append(trace_to_metrics_row("buy_and_hold", "oos", buy_hold_oos_trace, annualization=annualization))
    trades_frames.append(collect_trade_rows("buy_and_hold", "oos", buy_hold_oos_trace.trade_log))
    return metrics_rows, trades_frames, buy_hold_oos_trace
