from __future__ import annotations

import pandas as pd


TREE_SOURCE_COLUMNS: tuple[str, ...] = (
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
)

DEFAULT_TREE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "adx",
    "mfi",
    "ema_gap",
    "macd_gap",
    "di_gap",
    "kst_gap",
)


def build_feature_matrix(
    df: pd.DataFrame,
    signals: dict[str, pd.Series] | None = None,
    combined_signal: pd.Series | None = None,
) -> pd.DataFrame:
    missing = [c for c in TREE_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features for tree training: {missing}")

    x = pd.DataFrame(index=df.index)
    x["rsi"] = df["rsi"].astype(float)
    x["adx"] = df["adx"].astype(float)
    x["mfi"] = df["mfi"].astype(float)
    x["ema_gap"] = df["ema_fast"].astype(float) - df["ema_slow"].astype(float)
    x["macd_gap"] = df["macd"].astype(float) - df["macd_signal"].astype(float)
    x["di_gap"] = df["plus_di"].astype(float) - df["minus_di"].astype(float)
    x["kst_gap"] = df["kst"].astype(float) - df["kst_signal"].astype(float)

    if signals:
        signal_df = pd.DataFrame(index=df.index)
        for name, series in signals.items():
            signal_df[f"{name}_vote"] = series.reindex(df.index).fillna(0).astype("int8")
        x = x.join(signal_df)
        x["buy_vote_count"] = (signal_df == 1).sum(axis=1).astype("int8")
        x["sell_vote_count"] = (signal_df == -1).sum(axis=1).astype("int8")

    if combined_signal is not None:
        direction = combined_signal.reindex(df.index).fillna(0).astype("int8")
        x["signal_direction"] = direction
        x["signal_is_buy"] = (direction == 1).astype("int8")
        x["signal_is_sell"] = (direction == -1).astype("int8")

    return x
