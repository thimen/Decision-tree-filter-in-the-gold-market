from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from src.config import (
    BacktestConfig,
    ExperimentConfig,
    LabelConfig,
    MT5Config,
    MarketDataConfig,
    RunSpec,
    StrategyConfig,
    TreeConfig,
    thesis_run_spec,
)


MT5_ENV_OVERRIDES = {
    "login": "MT5_LOGIN",
    "password": "MT5_PASSWORD",
    "server": "MT5_SERVER",
}


def parse_int_grid(value: str) -> tuple[int, ...]:
    out = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not out:
        raise argparse.ArgumentTypeError("Grid must contain at least one integer")
    return out


def default_run_spec() -> RunSpec:
    return thesis_run_spec()


def _coerce_tree_fields(tree_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(tree_cfg)
    for key in ("max_depth_grid", "min_samples_leaf_grid"):
        if key not in out:
            continue
        value = out[key]
        if isinstance(value, str):
            out[key] = parse_int_grid(value)
        elif isinstance(value, (list, tuple)):
            out[key] = tuple(int(item) for item in value)
        else:
            raise ValueError(f"tree.{key} must be a list, tuple, or comma-separated string")
    return out


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _spec_from_dict(data: dict[str, Any]) -> RunSpec:
    allowed_top_level = {"results_dir", "experiment"}
    unknown_top_level = sorted(set(data.keys()) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"Unknown top-level config keys: {unknown_top_level}")

    base_spec = default_run_spec()
    merged = _merge_dicts(asdict(base_spec), data)
    experiment = merged["experiment"]
    experiment["tree"] = _coerce_tree_fields(experiment["tree"])

    return RunSpec(
        results_dir=str(merged["results_dir"]),
        experiment=ExperimentConfig(
            mt5=MT5Config(**experiment["mt5"]),
            market_data=MarketDataConfig(**experiment["market_data"]),
            strategy=StrategyConfig(**experiment["strategy"]),
            labels=LabelConfig(**experiment["labels"]),
            tree=TreeConfig(**experiment["tree"]),
            backtest=BacktestConfig(**experiment["backtest"]),
        ),
    )


def apply_mt5_env_overrides(spec: RunSpec) -> RunSpec:
    overrides: dict[str, Any] = {}
    for field_name, env_name in MT5_ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        overrides[field_name] = int(raw) if field_name == "login" else raw

    if not overrides:
        return spec

    return replace(
        spec,
        experiment=replace(
            spec.experiment,
            mt5=replace(spec.experiment.mt5, **overrides),
        ),
    )


def load_run_spec(path: str | Path) -> RunSpec:
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a JSON object")
    return apply_mt5_env_overrides(_spec_from_dict(raw))


def write_config_template(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(default_run_spec()), indent=2), encoding="utf-8")
    return output_path


def build_run_spec(args: argparse.Namespace) -> RunSpec:
    spec = load_run_spec(args.config) if args.config else default_run_spec()
    spec = apply_mt5_env_overrides(spec)

    experiment = spec.experiment
    mt5 = experiment.mt5
    market_data = experiment.market_data
    strategy = experiment.strategy
    labels = experiment.labels
    tree = experiment.tree
    backtest = experiment.backtest

    if args.results_dir is not None:
        spec = replace(spec, results_dir=args.results_dir)

    mt5_updates: dict[str, Any] = {}
    market_updates: dict[str, Any] = {}
    strategy_updates: dict[str, Any] = {}
    label_updates: dict[str, Any] = {}
    tree_updates: dict[str, Any] = {}
    backtest_updates: dict[str, Any] = {}

    for key, value in {
        "terminal_path": args.terminal_path,
        "login": args.login,
        "password": args.password,
        "server": args.server,
        "timeout_ms": args.timeout_ms,
        "portable": args.portable,
    }.items():
        if value is not None:
            mt5_updates[key] = value

    for key, value in {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "bars": args.bars,
        "utc_from": args.utc_from,
        "utc_to": args.utc_to,
    }.items():
        if value is not None:
            market_updates[key] = value

    for key, value in {
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "vote_k": args.vote_k,
        "oscillator_vote_mode": args.oscillator_vote_mode,
        "rsi_period": args.rsi_period,
        "rsi_buy": args.rsi_buy,
        "rsi_sell": args.rsi_sell,
        "macd_fast": args.macd_fast,
        "macd_slow": args.macd_slow,
        "macd_signal": args.macd_signal,
        "dmi_period": args.dmi_period,
        "mfi_period": args.mfi_period,
        "mfi_buy": args.mfi_buy,
        "mfi_sell": args.mfi_sell,
        "include_ema_vote": args.include_ema_vote,
        "kst_roc1": args.kst_roc1,
        "kst_roc2": args.kst_roc2,
        "kst_roc3": args.kst_roc3,
        "kst_roc4": args.kst_roc4,
        "kst_sma1": args.kst_sma1,
        "kst_sma2": args.kst_sma2,
        "kst_sma3": args.kst_sma3,
        "kst_sma4": args.kst_sma4,
        "kst_signal": args.kst_signal,
    }.items():
        if value is not None:
            strategy_updates[key] = value

    for key, value in {
        "horizon": args.label_horizon,
        "threshold": args.label_threshold,
    }.items():
        if value is not None:
            label_updates[key] = value

    for key, value in {
        "enabled": args.enable_tree_model,
        "max_depth_grid": args.tree_max_depth_grid,
        "min_samples_leaf_grid": args.tree_min_samples_leaf_grid,
        "min_training_samples": args.tree_min_training_samples,
        "random_state": args.tree_random_state,
        "holdout_ratio": args.tree_holdout_ratio,
        "time_series_splits": args.tree_time_series_splits,
        "confidence_bin_count": args.confidence_bin_count,
        "confidence_binning_mode": args.confidence_binning_mode,
    }.items():
        if value is not None:
            tree_updates[key] = value

    for key, value in {
        "cost_bps": args.cost_bps,
        "initial_cash": args.initial_cash,
        "trade_size_pct": args.trade_size_pct,
        "long_only": args.long_only,
        "take_profit_pct": args.take_profit_pct,
        "stop_loss_pct": args.stop_loss_pct,
    }.items():
        if value is not None:
            backtest_updates[key] = value

    if any((mt5_updates, market_updates, strategy_updates, label_updates, tree_updates, backtest_updates)):
        spec = replace(
            spec,
            experiment=replace(
                experiment,
                mt5=replace(mt5, **mt5_updates) if mt5_updates else mt5,
                market_data=replace(market_data, **market_updates) if market_updates else market_data,
                strategy=replace(strategy, **strategy_updates) if strategy_updates else strategy,
                labels=replace(labels, **label_updates) if label_updates else labels,
                tree=replace(tree, **tree_updates) if tree_updates else tree,
                backtest=replace(backtest, **backtest_updates) if backtest_updates else backtest,
            ),
        )

    return spec
