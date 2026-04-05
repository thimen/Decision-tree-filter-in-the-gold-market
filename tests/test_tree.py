from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tree import (
    assign_confidence_bins,
    build_feature_matrix,
    build_trade_outcome_labels,
    candidate_sort_key,
    export_tree_to_mql5,
    positive_class_probability,
    train_decision_tree,
)


def test_build_trade_outcome_labels_supports_long_and_short_candidates() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 106.0, 100.0, 94.0, 100.0],
            "high": [100.0, 106.5, 100.5, 100.5, 100.5],
            "low": [100.0, 99.5, 99.5, 93.5, 99.5],
        }
    )
    signal = pd.Series([1, 0, -1, 0, 0], dtype="int8")

    labels = build_trade_outcome_labels(
        df,
        signal=signal,
        take_profit_pct=0.06,
        stop_loss_pct=0.03,
        cost_bps=5.0,
        label_horizon=1,
        label_threshold=0.0,
    )

    assert labels.iloc[0] == 1.0
    assert labels.iloc[2] == 1.0


def test_build_feature_matrix_prefers_relative_and_vote_features() -> None:
    index = pd.RangeIndex(3)
    df = pd.DataFrame(
        {
            "ema_fast": [11.0, 12.0, 13.0],
            "ema_slow": [10.0, 10.0, 10.0],
            "rsi": [55.0, 60.0, 45.0],
            "macd": [1.0, 0.5, -0.5],
            "macd_signal": [0.2, 0.1, -0.2],
            "plus_di": [30.0, 25.0, 10.0],
            "minus_di": [10.0, 15.0, 25.0],
            "adx": [20.0, 30.0, 25.0],
            "kst": [2.0, 1.0, -1.0],
            "kst_signal": [0.5, 0.5, -0.5],
            "mfi": [60.0, 50.0, 40.0],
        },
        index=index,
    )
    signals = {"ema": pd.Series([1, 1, -1], index=index, dtype="int8")}
    combined = pd.Series([1, 0, -1], index=index, dtype="int8")

    features = build_feature_matrix(df, signals=signals, combined_signal=combined)

    assert {"rsi", "adx", "mfi", "ema_gap", "macd_gap", "di_gap", "kst_gap"}.issubset(features.columns)
    assert {"ema_vote", "buy_vote_count", "sell_vote_count", "signal_direction", "signal_is_buy", "signal_is_sell"}.issubset(
        features.columns
    )
    assert "ema_fast" not in features.columns
    assert "ema_slow" not in features.columns
    assert "volume" not in features.columns


def test_build_trade_outcome_labels_uses_new_entry_events_only() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 106.0, 107.0, 100.0],
            "high": [100.0, 106.5, 107.5, 100.5],
            "low": [100.0, 99.5, 106.5, 99.5],
        }
    )
    signal = pd.Series([1, 1, 0, 0], dtype="int8")

    labels = build_trade_outcome_labels(
        df,
        signal=signal,
        take_profit_pct=0.06,
        stop_loss_pct=0.03,
        cost_bps=5.0,
        label_horizon=2,
        label_threshold=0.0,
    )

    assert labels.iloc[0] == 1.0
    assert pd.isna(labels.iloc[1])


def test_build_trade_outcome_labels_marks_unresolved_trade_at_sample_end() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
        }
    )
    signal = pd.Series([1, 0, 0, 0], dtype="int8")

    labels = build_trade_outcome_labels(
        df,
        signal=signal,
        take_profit_pct=0.06,
        stop_loss_pct=0.03,
        cost_bps=5.0,
        label_horizon=None,
        label_threshold=0.0,
    )

    assert labels.iloc[0] == 1.0


def test_assign_confidence_bins_returns_ordered_ranges() -> None:
    proba = pd.Series([0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.40, 0.40, 0.50, 0.50], index=range(10))

    bins = assign_confidence_bins(proba, bin_count=5, mode="quantile")

    assert bins.labels.tolist() == [
        "range_1",
        "range_1",
        "range_2",
        "range_2",
        "range_3",
        "range_3",
        "range_4",
        "range_4",
        "range_5",
        "range_5",
    ]
    assert bins.bounds == [
        ("range_1", 0.1, 0.1),
        ("range_2", 0.2, 0.2),
        ("range_3", 0.3, 0.3),
        ("range_4", 0.4, 0.4),
        ("range_5", 0.5, 0.5),
    ]


def test_assign_confidence_bins_quantile_preserves_score_ties() -> None:
    proba = pd.Series([0.2] * 8 + [0.5] * 2, index=range(10))

    bins = assign_confidence_bins(proba, bin_count=5, mode="quantile")

    assert bins.labels.nunique() == 2
    assert bins.bounds == [
        ("range_1", 0.2, 0.2),
        ("range_2", 0.5, 0.5),
    ]
    assert set(bins.labels.iloc[:8]) == {"range_1"}
    assert set(bins.labels.iloc[8:]) == {"range_2"}


def test_assign_confidence_bins_fixed_width_preserves_score_ranges() -> None:
    proba = pd.Series([0.05, 0.22, 0.48, 0.76, 0.99], index=list("abcde"))

    bins = assign_confidence_bins(proba, bin_count=5, mode="fixed_width")

    assert bins.labels.tolist() == ["range_1", "range_2", "range_3", "range_4", "range_5"]
    assert bins.bounds[0] == ("range_1", 0.0, 0.2)


def test_candidate_sort_key_prefers_models_that_support_requested_band_count() -> None:
    supported = {"score": -0.30, "unique_probability_count": 5}
    unsupported = {"score": -0.10, "unique_probability_count": 4}

    assert candidate_sort_key(supported, minimum_unique_probabilities=5) > candidate_sort_key(
        unsupported,
        minimum_unique_probabilities=5,
    )


def test_train_decision_tree_uses_time_ordered_holdout_and_outputs_probabilities() -> None:
    index = pd.RangeIndex(120)
    direction = pd.Series([1 if i % 2 == 0 else -1 for i in index], index=index, dtype="int8")
    quality = pd.Series([1 if (i % 6) in (0, 1, 2) else 0 for i in index], index=index, dtype="int8")

    df = pd.DataFrame(
        {
            "ema_fast": 20.0 + quality,
            "ema_slow": 10.0,
            "rsi": 55.0 + (quality * 5.0),
            "macd": quality.astype(float),
            "macd_signal": 0.0,
            "plus_di": 25.0 + (quality * 10.0),
            "minus_di": 15.0 - (quality * 5.0),
            "adx": 20.0 + (quality * 5.0),
            "kst": quality.astype(float) * 2.0,
            "kst_signal": 0.0,
            "mfi": 55.0 + (quality * 5.0),
            "volume": 100.0,
            "tick_volume": 100.0,
            "spread": 10.0,
            "real_volume": 50.0,
        },
        index=index,
    )
    features = build_feature_matrix(df, combined_signal=direction)
    ml_df = features.join(pd.Series(quality, index=index, name="y"))

    result = train_decision_tree(
        ml_df=ml_df,
        max_depth_grid=(1, 2),
        min_samples_leaf_grid=(1, 5),
        random_state=42,
        holdout_ratio=0.2,
        time_series_splits=3,
    )

    assert result.validation_metric == "negative_brier_score"
    assert result.development_index[-1] < result.test_index[0]

    x_test = ml_df.loc[result.test_index].drop(columns=["y"])
    proba = positive_class_probability(result.model, x_test)
    assert len(proba) == len(result.test_index)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_export_tree_to_mql5_writes_probability_function(tmp_path: Path) -> None:
    index = pd.RangeIndex(20)
    df = pd.DataFrame(
        {
            "ema_fast": [2.0] * 20,
            "ema_slow": [1.0] * 20,
            "rsi": [60.0] * 20,
            "macd": [1.0] * 20,
            "macd_signal": [0.0] * 20,
            "plus_di": [30.0] * 20,
            "minus_di": [10.0] * 20,
            "adx": [25.0] * 20,
            "kst": [1.0] * 20,
            "kst_signal": [0.0] * 20,
            "mfi": [60.0] * 20,
        },
        index=index,
    )
    direction = pd.Series([1] * 20, index=index, dtype="int8")
    features = build_feature_matrix(df, combined_signal=direction)
    ml_df = features.join(pd.Series(([1] * 10) + ([0] * 10), index=index, name="y"))
    result = train_decision_tree(
        ml_df=ml_df,
        max_depth_grid=(1,),
        min_samples_leaf_grid=(1,),
        random_state=42,
        holdout_ratio=0.2,
        time_series_splits=3,
    )

    output_path = export_tree_to_mql5(result.model, tmp_path / "tree_gate_generated.mqh")
    text = output_path.read_text(encoding="utf-8")

    assert "EvaluateDecisionTreeProbability" in text
    assert "return" in text
    assert "ema_gap" in text
    assert "ema_fast" not in text
