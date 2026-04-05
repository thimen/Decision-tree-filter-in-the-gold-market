from __future__ import annotations

from dataclasses import dataclass, field


THESIS_UTC_FROM = "2019-12-17T12:00:00Z"
THESIS_UTC_TO = "2026-02-24T12:00:00Z"
THESIS_RESULTS_DIR = "results_thesis_us500_h1"
MARKET_DATA_TIMEZONE = "UTC"


@dataclass(frozen=True)
class MT5Config:
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 60_000
    portable: bool = False


@dataclass(frozen=True)
class MarketDataConfig:
    symbol: str
    timeframe: str = "H1"
    bars: int = 2_000
    utc_from: str | None = None
    utc_to: str | None = None


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int = 19
    ema_slow: int = 27
    vote_k: int = 3
    oscillator_vote_mode: str = "trend"
    rsi_period: int = 21
    rsi_buy: float = 45.0
    rsi_sell: float = 55.0
    macd_fast: int = 21
    macd_slow: int = 29
    macd_signal: int = 9
    dmi_period: int = 14
    mfi_period: int = 14
    mfi_buy: float = 45.0
    mfi_sell: float = 55.0
    include_ema_vote: bool = True
    kst_roc1: int = 10
    kst_roc2: int = 15
    kst_roc3: int = 20
    kst_roc4: int = 30
    kst_sma1: int = 10
    kst_sma2: int = 10
    kst_sma3: int = 10
    kst_sma4: int = 15
    kst_signal: int = 9


@dataclass(frozen=True)
class LabelConfig:
    horizon: int | None = None
    threshold: float = 0.0


@dataclass(frozen=True)
class TreeConfig:
    enabled: bool = False
    max_depth_grid: tuple[int, ...] = (2, 3, 4, 5, 6)
    min_samples_leaf_grid: tuple[int, ...] = (5, 10, 20, 50)
    min_training_samples: int = 60
    random_state: int = 42
    holdout_ratio: float = 0.2
    time_series_splits: int = 3
    confidence_bin_count: int = 5
    confidence_binning_mode: str = "quantile"


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 5.0
    initial_cash: float = 1.0
    trade_size_pct: float = 0.01
    long_only: bool = False
    take_profit_pct: float = 0.06
    stop_loss_pct: float = 0.03


@dataclass(frozen=True)
class ExperimentConfig:
    mt5: MT5Config
    market_data: MarketDataConfig
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass(frozen=True)
class RunSpec:
    experiment: ExperimentConfig
    results_dir: str = THESIS_RESULTS_DIR


def thesis_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        mt5=MT5Config(),
        market_data=MarketDataConfig(
            symbol="US500",
            timeframe="H1",
            bars=20_000,
            utc_from=THESIS_UTC_FROM,
            utc_to=THESIS_UTC_TO,
        ),
        strategy=StrategyConfig(),
        labels=LabelConfig(),
        tree=TreeConfig(enabled=True),
        backtest=BacktestConfig(),
    )


def thesis_run_spec() -> RunSpec:
    return RunSpec(experiment=thesis_experiment_config(), results_dir=THESIS_RESULTS_DIR)
