from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import ExperimentConfig
from src.run.artifacts import mask_series_outside_index
from src.run.suites import run_trace


def build_equity_frame(
    market: pd.DataFrame,
    baseline_signals: dict[str, pd.Series],
    full_traces: dict[str, Any],
    buy_hold_trace: Any,
    confidence_scores: pd.Series,
    confidence_labels: pd.Series,
    oos_market: pd.DataFrame,
    oos_index: pd.Index,
    cfg: ExperimentConfig,
    confidence_band_traces: dict[str, Any],
    multi_signal: pd.Series,
    buy_hold_oos_trace: Any,
    simulate_with_trace_fn: Callable[..., Any],
) -> pd.DataFrame:
    equity_df = market[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_volume",
            "spread",
            "real_volume",
            "ret",
            "ema_fast",
            "ema_slow",
            "rsi",
            "macd",
            "macd_signal",
            "plus_di",
            "minus_di",
            "adx",
            "kst",
            "kst_signal",
            "mfi",
        ]
    ].copy()
    for strategy_name, strategy_signal in baseline_signals.items():
        equity_df[f"signal_{strategy_name}"] = strategy_signal
        equity_df[f"equity_{strategy_name}_full"] = full_traces[strategy_name].result.equity
    equity_df["equity_buy_and_hold_full"] = buy_hold_trace.result.equity
    equity_df["confidence_score"] = confidence_scores
    equity_df["confidence_range"] = confidence_labels
    for strategy_name, strategy_signal in baseline_signals.items():
        oos_trace = run_trace(simulate_with_trace_fn, oos_market, strategy_signal.loc[oos_index], cfg.backtest)
        equity_df[f"equity_{strategy_name}_oos"] = mask_series_outside_index(oos_trace.result.equity, oos_index)
    equity_df["equity_buy_and_hold_oos"] = mask_series_outside_index(buy_hold_oos_trace.result.equity, oos_index)
    for strategy_name, trace in confidence_band_traces.items():
        equity_df[f"equity_{strategy_name}_oos"] = mask_series_outside_index(trace.result.equity, oos_index)
    return equity_df


def write_plot(
    out_dir: Path,
    oos_market: pd.DataFrame,
    oos_index: pd.Index,
    multi_signal: pd.Series,
    cfg: ExperimentConfig,
    confidence_band_traces: dict[str, Any],
    buy_hold_oos_trace: Any,
    simulate_with_trace_fn: Callable[..., Any],
) -> str:
    plot_status = "ok"
    try:
        from src.plots import plot_equity_curves

        oos_curves = {
            "multi_indicator_vote": mask_series_outside_index(
                run_trace(simulate_with_trace_fn, oos_market, multi_signal.loc[oos_index], cfg.backtest).result.equity,
                oos_index,
            ),
            "buy_and_hold": mask_series_outside_index(buy_hold_oos_trace.result.equity, oos_index),
        }
        for strategy_name, trace in confidence_band_traces.items():
            oos_curves[strategy_name] = mask_series_outside_index(trace.result.equity, oos_index)
        plot_equity_curves(oos_curves, out_dir / "equity_curves.png")
    except ModuleNotFoundError:
        plot_status = "skipped: matplotlib not installed"
    return plot_status
