from __future__ import annotations

import pandas as pd

from src.backtest import directional_return, entry_event_mask, resolve_exit


def build_trade_outcome_labels(
    df: pd.DataFrame,
    signal: pd.Series,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
    cost_bps: float,
    label_horizon: int | None = None,
    label_threshold: float = 0.0,
) -> pd.Series:
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe must contain {sorted(required)}")
    if take_profit_pct is None and stop_loss_pct is None:
        raise ValueError("At least one of take_profit_pct or stop_loss_pct must be set")
    if label_horizon is not None and int(label_horizon) <= 0:
        raise ValueError("label_horizon must be > 0 when set")

    aligned_signal = signal.reindex(df.index).fillna(0).astype("int8")
    aligned_entries = aligned_signal.where(entry_event_mask(aligned_signal), 0).astype("int8")
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    labels = pd.Series(index=df.index, dtype="float64")
    round_trip_cost = 2.0 * (float(cost_bps) / 10_000.0)

    for start in range(len(df)):
        direction = int(aligned_entries.iloc[start])
        if direction not in (-1, 1):
            continue

        entry_price = float(close.iloc[start])
        search_end = len(df) - 1
        if label_horizon is not None:
            search_end = start + int(label_horizon)
            if search_end >= len(df):
                continue

        label: float | None = None
        for end in range(start + 1, search_end + 1):
            exit_fill = resolve_exit(
                direction=direction,
                entry_price=entry_price,
                high=float(high.iloc[end]),
                low=float(low.iloc[end]),
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
            )
            if exit_fill is not None:
                _, exit_price = exit_fill
                net_result = directional_return(entry_price, exit_price, direction) - round_trip_cost
                label = 1.0 if net_result >= float(label_threshold) else 0.0
                break

        if label is None and search_end > start:
            realized_net = directional_return(entry_price, float(close.iloc[search_end]), direction) - round_trip_cost
            label = 1.0 if realized_net >= float(label_threshold) else 0.0

        if label is not None:
            labels.iloc[start] = label

    return labels
