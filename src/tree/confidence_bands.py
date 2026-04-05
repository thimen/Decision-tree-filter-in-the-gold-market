from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfidenceBins:
    labels: pd.Series
    bounds: list[tuple[str, float, float]]


def _assign_fixed_width_confidence_bins(positive_proba: pd.Series, bin_count: int) -> ConfidenceBins:
    if bin_count <= 0:
        raise ValueError("bin_count must be > 0")

    labels = [f"range_{idx + 1}" for idx in range(int(bin_count))]
    if len(positive_proba) == 0:
        return ConfidenceBins(
            labels=pd.Series(dtype="object", index=positive_proba.index),
            bounds=[],
        )

    counts = positive_proba.astype(float).value_counts(sort=False).sort_index()
    unique_scores = counts.index.to_numpy(dtype=float)
    total_count = int(counts.sum())
    requested_bin_count = int(bin_count)
    active_bin_count = min(requested_bin_count, len(unique_scores))
    active_labels = labels[:active_bin_count]

    score_to_label: dict[float, str] = {}
    interval_edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    cumulative = 0
    band_idx = 0
    band_start = 0
    target_counts = [total_count * (idx + 1) / active_bin_count for idx in range(active_bin_count)]

    for score_idx, score in enumerate(unique_scores):
        if band_idx < active_bin_count - 1:
            remaining_scores = len(unique_scores) - score_idx
            remaining_bands = active_bin_count - band_idx
            current_target = target_counts[band_idx]
            current_count = int(counts.iloc[score_idx])
            score_levels_in_band = score_idx - band_start
            can_close_current_band = score_levels_in_band >= 1 and remaining_scores >= remaining_bands
            if can_close_current_band:
                before_gap = abs(cumulative - current_target)
                after_gap = abs((cumulative + current_count) - current_target)
                if before_gap <= after_gap:
                    band_scores = unique_scores[band_start:score_idx]
                    label = active_labels[band_idx]
                    for band_score in band_scores:
                        score_to_label[float(band_score)] = label
                    band_idx += 1
                    band_start = score_idx

        cumulative += int(counts.iloc[score_idx])

    for idx in range(band_idx, active_bin_count):
        end = len(unique_scores) if idx == active_bin_count - 1 else band_start + 1
        band_scores = unique_scores[band_start:end]
        if len(band_scores) == 0:
            continue
        label = active_labels[idx]
        for band_score in band_scores:
            score_to_label[float(band_score)] = label
        band_start = end

    mapped = positive_proba.astype(float).map(score_to_label)
    bounds = [
        (labels[idx], float(interval_edges[idx]), float(interval_edges[idx + 1]))
        for idx in range(active_bin_count)
    ]
    return ConfidenceBins(labels=mapped.astype("object"), bounds=bounds)


def _assign_quantile_confidence_bins(positive_proba: pd.Series, bin_count: int) -> ConfidenceBins:
    if bin_count <= 0:
        raise ValueError("bin_count must be > 0")

    labels = [f"range_{idx + 1}" for idx in range(int(bin_count))]
    if len(positive_proba) == 0:
        return ConfidenceBins(
            labels=pd.Series(dtype="object", index=positive_proba.index),
            bounds=[],
        )

    counts = positive_proba.astype(float).value_counts(sort=False).sort_index()
    unique_scores = counts.index.to_numpy(dtype=float)
    active_bin_count = min(int(bin_count), len(unique_scores))
    active_labels = labels[:active_bin_count]
    total_count = int(counts.sum())
    score_to_label: dict[float, str] = {}
    bounds: list[tuple[str, float, float]] = []
    cumulative = 0
    band_idx = 0
    band_start = 0
    target_counts = [total_count * (idx + 1) / active_bin_count for idx in range(active_bin_count)]

    for score_idx, score in enumerate(unique_scores):
        if band_idx < active_bin_count - 1:
            remaining_scores = len(unique_scores) - score_idx
            remaining_bands = active_bin_count - band_idx
            current_target = target_counts[band_idx]
            current_count = int(counts.iloc[score_idx])
            score_levels_in_band = score_idx - band_start
            can_close_current_band = score_levels_in_band >= 1 and remaining_scores >= remaining_bands
            if can_close_current_band:
                before_gap = abs(cumulative - current_target)
                after_gap = abs((cumulative + current_count) - current_target)
                if before_gap <= after_gap:
                    band_scores = unique_scores[band_start:score_idx]
                    label = active_labels[band_idx]
                    for band_score in band_scores:
                        score_to_label[float(band_score)] = label
                    bounds.append((label, float(band_scores[0]), float(band_scores[-1])))
                    band_idx += 1
                    band_start = score_idx

        cumulative += int(counts.iloc[score_idx])

    for idx in range(band_idx, active_bin_count):
        end = len(unique_scores) if idx == active_bin_count - 1 else band_start + 1
        band_scores = unique_scores[band_start:end]
        if len(band_scores) == 0:
            continue
        label = active_labels[idx]
        for band_score in band_scores:
            score_to_label[float(band_score)] = label
        bounds.append((label, float(band_scores[0]), float(band_scores[-1])))
        band_start = end

    assigned = positive_proba.astype(float).map(score_to_label)
    return ConfidenceBins(labels=assigned.astype("object"), bounds=bounds)


def assign_confidence_bins(
    positive_proba: pd.Series,
    bin_count: int = 5,
    mode: str = "quantile",
) -> ConfidenceBins:
    mode_key = str(mode).strip().lower()
    if mode_key == "quantile":
        return _assign_quantile_confidence_bins(positive_proba, bin_count)
    if mode_key == "fixed_width":
        return _assign_fixed_width_confidence_bins(positive_proba, bin_count)
    raise ValueError("mode must be 'quantile' or 'fixed_width'")


def build_confidence_band_signal(
    signal: pd.Series,
    positive_proba: pd.Series,
    band_label: str,
    bin_labels: pd.Series,
    apply_on_index: pd.Index,
) -> pd.Series:
    del positive_proba
    filtered = pd.Series(0, index=signal.index, dtype="int8")
    accepted = bin_labels[(bin_labels == band_label) & bin_labels.index.isin(apply_on_index)].index
    filtered.loc[accepted] = signal.loc[accepted].astype("int8")
    return filtered
