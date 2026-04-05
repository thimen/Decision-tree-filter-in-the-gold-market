from __future__ import annotations

import pandas as pd


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def bull_regime(
    df: pd.DataFrame,
    require_slope_up: bool = False,
    slope_lookback: int = 3,
) -> pd.Series:
    regime = df["ema_fast"] > df["ema_slow"]
    if require_slope_up:
        regime = regime & (df["ema_slow"].diff(max(1, int(slope_lookback))) > 0)
    return regime.fillna(False)


def component_signals(
    df: pd.DataFrame,
    rsi_buy: float = 45.0,
    rsi_sell: float = 55.0,
    mfi_buy: float = 45.0,
    mfi_sell: float = 55.0,
    include_ema_vote: bool = True,
    oscillator_vote_mode: str = "trend",
) -> dict[str, pd.Series]:
    if float(rsi_buy) >= float(rsi_sell):
        raise ValueError("rsi_buy must be less than rsi_sell")
    if float(mfi_buy) >= float(mfi_sell):
        raise ValueError("mfi_buy must be less than mfi_sell")

    signals: dict[str, pd.Series] = {}

    if include_ema_vote:
        ema_signal = pd.Series(0, index=df.index, dtype="int8")
        ema_signal[df["ema_fast"] > df["ema_slow"]] = 1
        ema_signal[df["ema_fast"] < df["ema_slow"]] = -1
        signals["ema"] = ema_signal

    rsi_signal = pd.Series(0, index=df.index, dtype="int8")
    mfi_signal = pd.Series(0, index=df.index, dtype="int8")
    mode = oscillator_vote_mode.strip().lower()
    if mode == "mean_reversion":
        rsi_signal[df["rsi"] < float(rsi_buy)] = 1
        rsi_signal[df["rsi"] > float(rsi_sell)] = -1
        mfi_signal[df["mfi"] < float(mfi_buy)] = 1
        mfi_signal[df["mfi"] > float(mfi_sell)] = -1
    elif mode == "trend":
        rsi_signal[df["rsi"] > float(rsi_sell)] = 1
        rsi_signal[df["rsi"] < float(rsi_buy)] = -1
        mfi_signal[df["mfi"] > float(mfi_sell)] = 1
        mfi_signal[df["mfi"] < float(mfi_buy)] = -1
    else:
        raise ValueError("oscillator_vote_mode must be one of: trend, mean_reversion")

    macd_signal = pd.Series(0, index=df.index, dtype="int8")
    macd_signal[_cross_up(df["macd"], df["macd_signal"])] = 1
    macd_signal[_cross_down(df["macd"], df["macd_signal"])] = -1

    dmi_signal = pd.Series(0, index=df.index, dtype="int8")
    dmi_signal[df["plus_di"] > df["minus_di"]] = 1
    dmi_signal[df["plus_di"] < df["minus_di"]] = -1

    kst_signal = pd.Series(0, index=df.index, dtype="int8")
    kst_signal[_cross_up(df["kst"], df["kst_signal"])] = 1
    kst_signal[_cross_down(df["kst"], df["kst_signal"])] = -1

    signals["rsi"] = rsi_signal
    signals["macd"] = macd_signal
    signals["dmi"] = dmi_signal
    signals["kst"] = kst_signal
    signals["mfi"] = mfi_signal
    return signals


def combine_vote(
    signals: dict[str, pd.Series],
    vote_k: int = 3,
    bull: pd.Series | None = None,
) -> pd.Series:
    del bull
    if not signals:
        raise ValueError("No component signals provided")

    mat = pd.DataFrame(signals).fillna(0).astype("int8")
    buys = (mat == 1).sum(axis=1)
    sells = (mat == -1).sum(axis=1)

    out = pd.Series(0, index=mat.index, dtype="int8")
    out[(buys >= vote_k) & (buys > sells)] = 1
    out[(sells >= vote_k) & (sells > buys)] = -1
    return out
