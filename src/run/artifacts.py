from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.config import ExperimentConfig
from src.metrics import compute_performance


def config_to_safe_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    out = asdict(cfg)
    if out["mt5"]["password"]:
        out["mt5"]["password"] = "***"
    return out


def annualization_factor(timeframe: str) -> int:
    key = timeframe.strip().upper()
    if key == "D1":
        return 252
    if key == "W1":
        return 52
    if key == "MN1":
        return 12
    if key.startswith("H"):
        hours = int(key[1:])
        return int(252 * (24 / hours))
    if key.startswith("M"):
        minutes = int(key[1:])
        return int(252 * ((24 * 60) / minutes))
    raise ValueError(f"Unsupported timeframe for annualization: {timeframe}")


def mask_series_outside_index(series: pd.Series, index: pd.Index) -> pd.Series:
    masked = pd.Series(float("nan"), index=series.index, dtype="float64")
    masked.loc[index] = series.loc[index].astype(float)
    return masked


def trace_to_metrics_row(
    strategy: str,
    sample: str,
    trace: Any,
    annualization: int,
    confidence_range: str = "",
    confidence_min: float | None = None,
    confidence_max: float | None = None,
) -> dict[str, Any]:
    row = {
        "strategy": strategy,
        "sample": sample,
        "confidence_range": confidence_range,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
    }
    row.update(compute_performance(trace.result, trade_log=trace.trade_log, annualization=annualization))
    return row


def collect_trade_rows(strategy: str, sample: str, trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame(columns=["strategy", "sample"])
    out = trade_log.copy()
    out.insert(0, "sample", sample)
    out.insert(0, "strategy", strategy)
    return out
