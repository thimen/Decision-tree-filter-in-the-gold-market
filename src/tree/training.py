from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class TreeSearchResult:
    model: DecisionTreeClassifier
    best_params: dict[str, int]
    best_cv_score: float
    validation_metric: str
    development_index: pd.Index
    test_index: pd.Index
    unique_probability_count: int
    requested_unique_probability_count: int
    satisfied_unique_probability_requirement: bool


def split_development_test(index: pd.Index, holdout_ratio: float = 0.2) -> tuple[pd.Index, pd.Index]:
    if len(index) < 2:
        raise ValueError("Need at least two samples for development/test split")
    holdout_count = max(1, int(np.ceil(len(index) * float(holdout_ratio))))
    if holdout_count >= len(index):
        holdout_count = 1
    split_at = len(index) - holdout_count
    return index[:split_at], index[split_at:]


def probability_score(y_true: pd.Series, positive_proba: pd.Series) -> float:
    aligned = y_true.astype(float).reindex(positive_proba.index)
    if len(aligned) == 0:
        return float("-inf")
    brier = float(((aligned - positive_proba) ** 2).mean())
    return -brier


def candidate_sort_key(candidate: dict[str, Any], minimum_unique_probabilities: int) -> tuple[int, float, int]:
    meets_requirement = int(candidate["unique_probability_count"] >= minimum_unique_probabilities)
    return (meets_requirement, float(candidate["score"]), int(candidate["unique_probability_count"]))


def train_decision_tree(
    ml_df: pd.DataFrame,
    max_depth_grid: tuple[int, ...],
    min_samples_leaf_grid: tuple[int, ...],
    random_state: int,
    holdout_ratio: float = 0.2,
    time_series_splits: int = 3,
    minimum_unique_probabilities: int = 1,
) -> TreeSearchResult:
    development_index, test_index = split_development_test(ml_df.index, holdout_ratio=holdout_ratio)
    if len(development_index) < 2 or len(test_index) == 0:
        raise ValueError("Insufficient samples for thesis-aligned development/test split")

    x_dev = ml_df.loc[development_index].drop(columns=["y"])
    y_dev = ml_df.loc[development_index, "y"].astype(int)

    split_count = min(int(time_series_splits), max(2, len(x_dev) - 1))
    tscv = TimeSeriesSplit(n_splits=split_count)

    candidate_results: list[dict[str, Any]] = []

    valid_pairs = [(int(d), int(l)) for d, l in product(max_depth_grid, min_samples_leaf_grid) if int(l) <= len(x_dev)]
    if not valid_pairs:
        valid_pairs = [(int(max_depth_grid[0]), 1)]

    for max_depth, min_leaf in valid_pairs:
        fold_scores: list[float] = []
        for train_pos, val_pos in tscv.split(x_dev):
            x_train = x_dev.iloc[train_pos]
            y_train = y_dev.iloc[train_pos]
            x_val = x_dev.iloc[val_pos]
            y_val = y_dev.iloc[val_pos]
            if y_train.nunique() < 2:
                continue
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=random_state,
            )
            model.fit(x_train, y_train)
            proba = positive_class_probability(model, x_val)
            fold_scores.append(probability_score(y_val, proba))

        if not fold_scores:
            continue

        mean_score = float(np.mean(fold_scores))
        candidate_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=random_state,
        )
        candidate_model.fit(x_dev, y_dev)
        unique_probability_count = int(positive_class_probability(candidate_model, x_dev).nunique())
        candidate_results.append(
            {
                "params": {"max_depth": int(max_depth), "min_samples_leaf": int(min_leaf)},
                "score": mean_score,
                "unique_probability_count": unique_probability_count,
            }
        )

    if not candidate_results:
        raise ValueError("Decision tree search did not yield a valid model")

    requested_unique_probability_count = max(1, int(minimum_unique_probabilities))
    best_candidate = max(
        candidate_results,
        key=lambda candidate: candidate_sort_key(candidate, requested_unique_probability_count),
    )
    best_params = dict(best_candidate["params"])

    final_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=random_state,
    )
    final_model.fit(x_dev, y_dev)

    return TreeSearchResult(
        model=final_model,
        best_params=best_params,
        best_cv_score=float(best_candidate["score"]),
        validation_metric="negative_brier_score",
        development_index=development_index,
        test_index=test_index,
        unique_probability_count=int(best_candidate["unique_probability_count"]),
        requested_unique_probability_count=requested_unique_probability_count,
        satisfied_unique_probability_requirement=bool(
            int(best_candidate["unique_probability_count"]) >= requested_unique_probability_count
        ),
    )


def positive_class_probability(model: DecisionTreeClassifier, x: pd.DataFrame) -> pd.Series:
    if len(x) == 0:
        return pd.Series(dtype="float64", index=x.index)

    class_lookup = {int(cls): i for i, cls in enumerate(model.classes_)}
    if 1 not in class_lookup:
        return pd.Series(0.0, index=x.index, dtype="float64")
    if len(class_lookup) == 1:
        value = 1.0 if int(model.classes_[0]) == 1 else 0.0
        return pd.Series(value, index=x.index, dtype="float64")
    proba = model.predict_proba(x)[:, class_lookup[1]]
    return pd.Series(proba, index=x.index, dtype="float64")
