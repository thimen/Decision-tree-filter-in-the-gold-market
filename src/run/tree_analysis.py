from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import ExperimentConfig
from src.run.artifacts import collect_trade_rows, trace_to_metrics_row
from src.run.suites import run_trace
from src.tree import (
    assign_confidence_bins,
    build_confidence_band_signal,
    build_feature_matrix,
    build_trade_outcome_labels,
    export_tree_to_mql5,
    positive_class_probability,
    split_development_test,
    train_decision_tree,
)


def run_tree_suite(
    market: pd.DataFrame,
    components: dict[str, pd.Series],
    multi_signal: pd.Series,
    cfg: ExperimentConfig,
    out_dir: Path,
    annualization: int,
    simulate_with_trace_fn: Callable[..., Any],
) -> tuple[pd.Index, dict[str, Any], dict[str, Any], pd.Series, pd.Series, str, str, list[dict[str, Any]], list[pd.DataFrame]]:
    _, default_market_test_index = split_development_test(market.index, holdout_ratio=cfg.tree.holdout_ratio)
    oos_index = default_market_test_index
    tree_gate_path = ""
    confidence_scores_path = ""
    tree_metadata: dict[str, Any] = {
        "enabled": bool(cfg.tree.enabled),
        "skipped": not bool(cfg.tree.enabled),
        "validation_metric": "not_run",
    }
    confidence_band_traces: dict[str, Any] = {}
    confidence_labels = pd.Series("", index=market.index, dtype="object")
    confidence_scores = pd.Series(float("nan"), index=market.index, dtype="float64")
    metrics_rows: list[dict[str, Any]] = []
    trades_frames: list[pd.DataFrame] = []

    if not cfg.tree.enabled:
        return (
            oos_index,
            tree_metadata,
            confidence_band_traces,
            confidence_labels,
            confidence_scores,
            tree_gate_path,
            confidence_scores_path,
            metrics_rows,
            trades_frames,
        )

    features = build_feature_matrix(market, components, combined_signal=multi_signal)
    labels = build_trade_outcome_labels(
        market,
        signal=multi_signal,
        take_profit_pct=cfg.backtest.take_profit_pct,
        stop_loss_pct=cfg.backtest.stop_loss_pct,
        cost_bps=cfg.backtest.cost_bps,
        label_horizon=cfg.labels.horizon,
        label_threshold=cfg.labels.threshold,
    )
    candidate_mask = multi_signal != 0
    ml_df = features.loc[candidate_mask].join(labels.rename("y")).dropna()

    if len(ml_df) < cfg.tree.min_training_samples or ml_df["y"].nunique() <= 1:
        tree_metadata = {
            "enabled": True,
            "skipped": True,
            "validation_metric": "not_run",
            "reason": "Too few labeled candidate samples for tree training",
            "ml_samples": int(len(ml_df)),
            "min_training_samples": int(cfg.tree.min_training_samples),
        }
        return (
            oos_index,
            tree_metadata,
            confidence_band_traces,
            confidence_labels,
            confidence_scores,
            tree_gate_path,
            confidence_scores_path,
            metrics_rows,
            trades_frames,
        )

    tree_result = train_decision_tree(
        ml_df=ml_df,
        max_depth_grid=cfg.tree.max_depth_grid,
        min_samples_leaf_grid=cfg.tree.min_samples_leaf_grid,
        random_state=cfg.tree.random_state,
        holdout_ratio=cfg.tree.holdout_ratio,
        time_series_splits=cfg.tree.time_series_splits,
        minimum_unique_probabilities=cfg.tree.confidence_bin_count,
    )
    oos_start = tree_result.test_index[0]
    oos_index = market.index[market.index >= oos_start]
    x_test = ml_df.loc[tree_result.test_index].drop(columns=["y"])
    y_test = ml_df.loc[tree_result.test_index, "y"].astype(int)
    proba_test = positive_class_probability(tree_result.model, x_test)
    bins = assign_confidence_bins(
        proba_test,
        bin_count=cfg.tree.confidence_bin_count,
        mode=cfg.tree.confidence_binning_mode,
    )
    confidence_scores.loc[proba_test.index] = proba_test
    confidence_labels.loc[bins.labels.index] = bins.labels

    confidence_scores_df = pd.DataFrame(
        {
            "time": proba_test.index,
            "signal_direction": multi_signal.loc[proba_test.index].astype("int8").values,
            "confidence_score": proba_test.values,
            "confidence_range": bins.labels.values,
            "label": y_test.values,
        }
    )
    confidence_scores_path = str(out_dir / "confidence_scores.csv")
    confidence_scores_df.to_csv(confidence_scores_path, index=False)

    test_market = market.loc[oos_index]
    for band_label, band_min, band_max in bins.bounds:
        band_signal = build_confidence_band_signal(
            signal=multi_signal,
            positive_proba=proba_test,
            band_label=band_label,
            bin_labels=bins.labels,
            apply_on_index=proba_test.index,
        )
        trace = run_trace(simulate_with_trace_fn, test_market, band_signal.loc[oos_index], cfg.backtest)
        strategy_name = f"tree_confidence_{band_label}"
        confidence_band_traces[strategy_name] = trace
        metrics_rows.append(
            trace_to_metrics_row(
                strategy=strategy_name,
                sample="oos",
                trace=trace,
                annualization=annualization,
                confidence_range=band_label,
                confidence_min=band_min,
                confidence_max=band_max,
            )
        )
        trades_frames.append(collect_trade_rows(strategy_name, "oos", trace.trade_log))

    tree_gate_path = str(export_tree_to_mql5(tree_result.model, out_dir / "tree_gate_generated.mqh"))
    tree_metadata = {
        "enabled": True,
        "skipped": False,
        "validation_metric": tree_result.validation_metric,
        "best_params": tree_result.best_params,
        "best_cv_score": tree_result.best_cv_score,
        "development_unique_probability_count": int(tree_result.unique_probability_count),
        "requested_unique_probability_count": int(tree_result.requested_unique_probability_count),
        "satisfied_unique_probability_requirement": bool(tree_result.satisfied_unique_probability_requirement),
        "development_samples": int(len(tree_result.development_index)),
        "test_candidate_samples": int(len(tree_result.test_index)),
        "min_training_samples": int(cfg.tree.min_training_samples),
        "oos_start": str(oos_index[0]),
        "oos_end": str(oos_index[-1]),
        "confidence_bin_count": int(cfg.tree.confidence_bin_count),
        "confidence_binning_mode": str(cfg.tree.confidence_binning_mode),
        "active_confidence_band_count": int(len(bins.bounds)),
        "tree_gate_generated_path": tree_gate_path,
    }
    return (
        oos_index,
        tree_metadata,
        confidence_band_traces,
        confidence_labels,
        confidence_scores,
        tree_gate_path,
        confidence_scores_path,
        metrics_rows,
        trades_frames,
    )
