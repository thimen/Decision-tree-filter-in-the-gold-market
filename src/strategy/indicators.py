from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.clip(0, 100)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line})


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr_components = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    tr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0.0, np.nan))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0.0, np.nan))
    adx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    raw_flow = typical_price * volume
    prev_tp = typical_price.shift(1)
    pos_flow = raw_flow.where(typical_price > prev_tp, 0.0)
    neg_flow = raw_flow.where(typical_price < prev_tp, 0.0)
    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    value = 100 - (100 / (1 + ratio))
    return value.clip(0, 100)


def _roc(close: pd.Series, period: int) -> pd.Series:
    base = close.shift(period)
    return ((close - base) / base.replace(0.0, np.nan)) * 100.0


def _kst(
    close: pd.Series,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal: int = 9,
) -> pd.DataFrame:
    rcma1 = _roc(close, roc1).rolling(sma1, min_periods=sma1).mean()
    rcma2 = _roc(close, roc2).rolling(sma2, min_periods=sma2).mean()
    rcma3 = _roc(close, roc3).rolling(sma3, min_periods=sma3).mean()
    rcma4 = _roc(close, roc4).rolling(sma4, min_periods=sma4).mean()
    kst = rcma1 + (2.0 * rcma2) + (3.0 * rcma3) + (4.0 * rcma4)
    kst_signal = kst.rolling(signal, min_periods=signal).mean()
    return pd.DataFrame({"kst": kst, "kst_signal": kst_signal})


def add_multi_indicator_columns(
    df: pd.DataFrame,
    ema_fast: int = 19,
    ema_slow: int = 27,
    rsi_period: int = 21,
    macd_fast: int = 21,
    macd_slow: int = 29,
    macd_signal: int = 9,
    dmi_period: int = 14,
    mfi_period: int = 14,
    kst_roc1: int = 10,
    kst_roc2: int = 15,
    kst_roc3: int = 20,
    kst_roc4: int = 30,
    kst_sma1: int = 10,
    kst_sma2: int = 10,
    kst_sma3: int = 10,
    kst_sma4: int = 15,
    kst_signal: int = 9,
) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=ema_slow, adjust=False).mean()
    out["rsi"] = _rsi(out["close"], period=rsi_period)

    out = out.join(_macd(out["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal))
    out = out.join(_dmi(out["high"], out["low"], out["close"], period=dmi_period))
    out["mfi"] = _mfi(out["high"], out["low"], out["close"], out["volume"], period=mfi_period)
    out = out.join(
        _kst(
            out["close"],
            roc1=kst_roc1,
            roc2=kst_roc2,
            roc3=kst_roc3,
            roc4=kst_roc4,
            sma1=kst_sma1,
            sma2=kst_sma2,
            sma3=kst_sma3,
            sma4=kst_sma4,
            signal=kst_signal,
        )
    )
    return out
