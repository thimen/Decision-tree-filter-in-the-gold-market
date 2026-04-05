from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import ExperimentConfig, MARKET_DATA_TIMEZONE
from src.run.artifacts import annualization_factor, config_to_safe_dict
from src.run.market import fetch_market, prepare_signals
from src.run.output_files import build_equity_frame, write_plot
from src.run.suites import run_full_sample_suite, run_oos_baselines
from src.run.tree_analysis import run_tree_suite


def execute_experiment(
    cfg: ExperimentConfig,
    results_dir: str,
    *,
    mt5_module: Any | None,
    parse_utc_fn: Callable[[str | None], Any],
    add_multi_indicator_columns_fn: Callable[..., pd.DataFrame],
    component_signals_fn: Callable[..., dict[str, pd.Series]],
    combine_vote_fn: Callable[..., pd.Series],
    simulate_with_trace_fn: Callable[..., Any],
) -> dict[str, str]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    market = fetch_market(cfg, mt5_module, parse_utc_fn)
    market, components, multi_signal, baseline_signals = prepare_signals(
        market,
        cfg,
        add_multi_indicator_columns_fn=add_multi_indicator_columns_fn,
        component_signals_fn=component_signals_fn,
        combine_vote_fn=combine_vote_fn,
    )

    annualization = annualization_factor(cfg.market_data.timeframe)
    full_metrics, full_trades, full_traces, buy_hold_trace = run_full_sample_suite(
        market,
        baseline_signals,
        cfg,
        annualization,
        simulate_with_trace_fn,
    )

    (
        oos_index,
        tree_metadata,
        confidence_band_traces,
        confidence_labels,
        confidence_scores,
        tree_gate_path,
        confidence_scores_path,
        tree_metrics,
        tree_trades,
    ) = run_tree_suite(
        market,
        components,
        multi_signal,
        cfg,
        out_dir,
        annualization,
        simulate_with_trace_fn,
    )

    oos_market = market.loc[oos_index]
    oos_metrics, oos_trades, buy_hold_oos_trace = run_oos_baselines(
        oos_market,
        oos_index,
        baseline_signals,
        cfg,
        annualization,
        simulate_with_trace_fn,
    )

    metrics_rows = full_metrics + tree_metrics + oos_metrics
    trades_frames = full_trades + tree_trades + oos_trades

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    trade_rows = [df.dropna(axis=1, how="all") for df in trades_frames if not df.empty]
    combined_trades = pd.concat(trade_rows, ignore_index=True, sort=False) if trade_rows else pd.DataFrame()
    trades_path = out_dir / "trades.csv"
    combined_trades.to_csv(trades_path, index=False)

    equity_df = build_equity_frame(
        market,
        baseline_signals,
        full_traces,
        buy_hold_trace,
        confidence_scores,
        confidence_labels,
        oos_market,
        oos_index,
        cfg,
        confidence_band_traces,
        multi_signal,
        buy_hold_oos_trace,
        simulate_with_trace_fn,
    )
    equity_path = out_dir / "equity_curves.csv"
    equity_df.to_csv(equity_path)

    rates_path = out_dir / "rates.csv"
    market[["open", "high", "low", "close", "volume", "tick_volume", "spread", "real_volume", "ret"]].to_csv(rates_path)

    plot_status = write_plot(
        out_dir,
        oos_market,
        oos_index,
        multi_signal,
        cfg,
        confidence_band_traces,
        buy_hold_oos_trace,
        simulate_with_trace_fn,
    )

    config_path = out_dir / "config_used.json"
    with config_path.open("w", encoding="utf-8") as handle:
        cfg_out = config_to_safe_dict(cfg)
        cfg_out["tree_training"] = tree_metadata
        cfg_out["metrics_annualization_factor"] = annualization
        cfg_out["market_data_timezone"] = MARKET_DATA_TIMEZONE
        cfg_out["oos_start"] = str(oos_index[0])
        cfg_out["oos_end"] = str(oos_index[-1])
        json.dump(cfg_out, handle, indent=2)

    return {
        "metrics_path": str(metrics_path),
        "equity_curves_path": str(equity_path),
        "rates_path": str(rates_path),
        "trades_path": str(trades_path),
        "config_path": str(config_path),
        "tree_gate_path": tree_gate_path,
        "confidence_scores_path": confidence_scores_path,
        "plot_status": plot_status,
    }
