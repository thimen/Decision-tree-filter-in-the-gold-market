# Thesis MT5 Experiment

This repository follows this flow:

1. Pull `XAUUSD` `H1` data from MetaTrader 5.
2. Preserve OHLC, tick volume, spread, and real volume.
3. Build `EMA`, `RSI`, `MACD`, `DMI`, `KST`, and `MFI`.
4. Generate discrete `buy`, `sell`, and `hold` rule signals.
5. Train a decision tree as a confidence-scoring model on candidate multi-indicator signals.
6. Evaluate single-indicator baselines, the multi-indicator baseline, and out-of-sample confidence quantiles.
7. Export the tree gate as readable MQL5 logic for MT5 Strategy Tester use.

## Main Files

- [src/experiment.py](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/src/experiment.py): thesis-aligned experiment entrypoint.
- [src/strategy/indicators.py](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/src/strategy/indicators.py): deterministic feature engineering for thesis indicators.
- [src/strategy/signals.py](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/src/strategy/signals.py): discrete indicator votes and multi-indicator aggregation.
- [src/backtest.py](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/src/backtest.py): directional backtest with `buy/sell/hold` semantics and TP/SL exits.
- [src/tree/__init__.py](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/src/tree/__init__.py): single public entrypoint for chronological tree training, confidence bins, and MQL export.
- [src/config.py](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/src/config.py): canonical thesis defaults and the typed run specification.
- [mql5/Experts/DecisionTreeMultiIndicatorEA.mq5](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/mql5/Experts/DecisionTreeMultiIndicatorEA.mq5): optional MT5 execution scaffold using the exported tree gate.
- [configs/run_config.xauusd_h1.json](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/configs/run_config.xauusd_h1.json): thesis reproduction config.
- [configs/run_config.xauusd_h4.json](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/configs/run_config.xauusd_h4.json): optional XAUUSD H4 comparison config.

## Code Layout

- `src/experiment.py`: CLI entrypoint and run assembly only.
- `src/run/`: experiment orchestration, flat-config loading, and output generation.
- `src/strategy/`: indicator feature generation and rule-signal construction.
- `src/tree/`: candidate labeling, feature selection, tree training, confidence bands, and MQL export.
- `src/backtest.py`: trade execution engine used by all strategy suites.

## Install

```bash
pip install -e .
```

On macOS, use the Windows Python inside the same Wine prefix as MT5 for real data runs. The default configs leave `terminal_path` as `null` so MetaTrader5 can attach to the already running terminal in that prefix instead of forcing a separate install path.

## Thesis Run

```bash
python -m src.experiment --config configs/run_config.xauusd_h1.json
```

Wine example:

```bash
"$WINE10" "C:\Python311\python.exe" -m src.experiment --config "Z:/Users/thimthor/Desktop/skola/Examensarbete/Kod1/configs/run_config.xauusd_h1.json"
```

Quick connection probe:

```bash
"$WINE10" "C:\Python311\python.exe" -m src.check_mt5_connection --config "Z:/Users/thimthor/Desktop/skola/Examensarbete/Kod1/configs/run_config.xauusd_h1.json"
```

If MT5 on your machine does not let the Python bridge reuse the GUI login, set the broker account credentials through environment variables:

```bash
export MT5_LOGIN=12345678
export MT5_PASSWORD='your_trade_account_password'
export MT5_SERVER='YourBroker-Server'
```

The config loader reads `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER` automatically. Environment variables override the JSON config, and CLI flags override both. Thesis-facing configs in this repo keep broker credentials as `null` on purpose.

Config files now mirror the typed `RunSpec` structure directly: top-level `results_dir`, then nested `experiment.mt5`, `experiment.market_data`, `experiment.strategy`, `experiment.labels`, `experiment.tree`, and `experiment.backtest`.

There is also an example file at [.env.example](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/.env.example). To use it in your shell:

```bash
cp .env.example .env
source .env
```

## Outputs

- `metrics.csv`: full-sample and out-of-sample metrics for single-indicator, multi-indicator, and confidence-range strategies.
- `confidence_scores.csv`: held-out candidate entry events with decision-tree probabilities and quantile-based confidence ranges.
- `equity_curves.csv`: thesis dataset fields, signals, confidence annotations, and equity series.
- `trades.csv`: combined trade log across reported strategies.
- `rates.csv`: MT5 market data with spread and both volume fields preserved.
- `config_used.json`: full run configuration and tree-training metadata.
- `config_used.json` also records the market timestamp timezone (`UTC`) and the minimum sample requirement for tree training.
- `tree_gate_generated.mqh`: exported decision-tree probability function for MQL5. Copy this over [mql5/Include/TreeGate.mqh](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/mql5/Include/TreeGate.mqh) before compiling the EA.

The MT5 EA scaffold assumes a hedging account so multiple thesis trades can coexist independently.

## Tests

Use the repo virtual environment:

```bash
.venv/bin/python -m pytest -q
```

The test suite covers:

- MT5 data normalization with thesis fields
- directional `buy/sell/hold` backtesting semantics
- long and short TP/SL labeling
- time-ordered decision-tree validation
- confidence-bin assignment
- thesis-aligned end-to-end experiment output

## Reproducibility

- Time handling: MT5 timestamps are stored and exported as UTC, and `config_used.json` records `market_data_timezone` as `UTC`.
- Authentication: thesis configs never store broker credentials; use `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER` from the shell environment instead.
- Tree gating: the minimum candidate-sample requirement for tree training is explicit in config as `experiment.tree.min_training_samples` and is copied into `config_used.json`.
- Artifact regeneration: rerun the same config file to regenerate `metrics.csv`, `confidence_scores.csv`, `equity_curves.csv`, `trades.csv`, `rates.csv`, `config_used.json`, and `tree_gate_generated.mqh` for that experiment.
