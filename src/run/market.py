from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.config import ExperimentConfig
from src.mt5_client import MT5Client


def fetch_market(
    cfg: ExperimentConfig,
    mt5_module: Any | None,
    parse_utc_fn: Callable[[str | None], Any],
) -> pd.DataFrame:
    with MT5Client(
        terminal_path=cfg.mt5.terminal_path,
        login=cfg.mt5.login,
        password=cfg.mt5.password,
        server=cfg.mt5.server,
        timeout_ms=cfg.mt5.timeout_ms,
        portable=cfg.mt5.portable,
        mt5_module=mt5_module,
    ) as client:
        return client.fetch_rates(
            symbol=cfg.market_data.symbol,
            timeframe=cfg.market_data.timeframe,
            bars=cfg.market_data.bars,
            utc_from=parse_utc_fn(cfg.market_data.utc_from),
            utc_to=parse_utc_fn(cfg.market_data.utc_to),
        )


def prepare_signals(
    market: pd.DataFrame,
    cfg: ExperimentConfig,
    add_multi_indicator_columns_fn: Callable[..., pd.DataFrame],
    component_signals_fn: Callable[..., dict[str, pd.Series]],
    combine_vote_fn: Callable[..., pd.Series],
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series, dict[str, pd.Series]]:
    enriched_market = add_multi_indicator_columns_fn(
        market,
        ema_fast=cfg.strategy.ema_fast,
        ema_slow=cfg.strategy.ema_slow,
        rsi_period=cfg.strategy.rsi_period,
        macd_fast=cfg.strategy.macd_fast,
        macd_slow=cfg.strategy.macd_slow,
        macd_signal=cfg.strategy.macd_signal,
        dmi_period=cfg.strategy.dmi_period,
        mfi_period=cfg.strategy.mfi_period,
        kst_roc1=cfg.strategy.kst_roc1,
        kst_roc2=cfg.strategy.kst_roc2,
        kst_roc3=cfg.strategy.kst_roc3,
        kst_roc4=cfg.strategy.kst_roc4,
        kst_sma1=cfg.strategy.kst_sma1,
        kst_sma2=cfg.strategy.kst_sma2,
        kst_sma3=cfg.strategy.kst_sma3,
        kst_sma4=cfg.strategy.kst_sma4,
        kst_signal=cfg.strategy.kst_signal,
    )
    components = component_signals_fn(
        enriched_market,
        rsi_buy=cfg.strategy.rsi_buy,
        rsi_sell=cfg.strategy.rsi_sell,
        mfi_buy=cfg.strategy.mfi_buy,
        mfi_sell=cfg.strategy.mfi_sell,
        include_ema_vote=cfg.strategy.include_ema_vote,
        oscillator_vote_mode=cfg.strategy.oscillator_vote_mode,
    )
    multi_signal = combine_vote_fn(components, vote_k=cfg.strategy.vote_k)
    if cfg.backtest.long_only:
        multi_signal = multi_signal.clip(lower=0)
        components = {name: series.clip(lower=0).astype("int8") for name, series in components.items()}

    baseline_signals: dict[str, pd.Series] = {f"single_{name}": series.astype("int8") for name, series in components.items()}
    baseline_signals["multi_indicator_vote"] = multi_signal.astype("int8")
    return enriched_market, components, multi_signal, baseline_signals
