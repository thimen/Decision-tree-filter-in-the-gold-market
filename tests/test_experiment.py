from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.config import thesis_run_spec
from src.experiment import run_experiment
from src.run.artifacts import annualization_factor
from src.run.configuration import load_run_spec, write_config_template


class PatternMT5:
    TIMEFRAME_H1 = 60

    def __init__(self) -> None:
        self.connected = False

    def initialize(self, **kwargs):
        del kwargs
        self.connected = True
        return True

    def shutdown(self):
        self.connected = False

    def last_error(self):
        return (1, "error")

    def symbol_select(self, symbol, enabled):
        return enabled and symbol == "US500"

    def _rows(self, bars: int) -> list[dict[str, float]]:
        base_time = 1704067200
        rows: list[dict[str, float]] = []
        for i in range(bars):
            close = 100.0
            high = 100.5
            low = 99.5
            spread = 12
            if i > 0 and (i - 1) % 3 == 0:
                trade_id = (i - 1) // 3
                direction = 1 if trade_id % 2 == 0 else -1
                good = (trade_id % 4) in (0, 1)
                if direction == 1:
                    if good:
                        close, high, low = 106.0, 106.5, 99.0
                    else:
                        close, high, low = 97.0, 101.0, 96.5
                else:
                    if good:
                        close, high, low = 94.0, 100.5, 93.5
                    else:
                        close, high, low = 103.0, 103.5, 99.0
            rows.append(
                {
                    "time": base_time + i * 3600,
                    "open": 100.0,
                    "high": high,
                    "low": low,
                    "close": close,
                    "tick_volume": 100 + i,
                    "spread": spread,
                    "real_volume": 50 + i,
                }
            )
        return rows

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, bars):
        del symbol, timeframe, start_pos
        return self._rows(bars)

    def copy_rates_range(self, symbol, timeframe, utc_from, utc_to):
        del symbol, timeframe, utc_from, utc_to
        return self._rows(360)


def test_run_experiment_writes_thesis_aligned_outputs(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "results"

    def fake_add_multi_indicator_columns(df, **kwargs):
        del kwargs
        out = df.copy()
        idx = pd.Series(range(len(out)), index=out.index)
        trade_id = idx // 3
        is_candidate = idx % 3 == 0
        direction = pd.Series(0, index=out.index, dtype="int8")
        direction.loc[is_candidate & ((trade_id % 2) == 0)] = 1
        direction.loc[is_candidate & ((trade_id % 2) == 1)] = -1
        quality = ((trade_id % 4).isin([0, 1])).astype(int)

        out["ema_fast"] = 20.0 + (quality * 5.0)
        out["ema_slow"] = 10.0
        out["rsi"] = 50.0 + (quality * 15.0)
        out["macd"] = quality.astype(float)
        out["macd_signal"] = 0.0
        out["plus_di"] = 15.0 + (quality * 20.0)
        out["minus_di"] = 25.0 - (quality * 10.0)
        out["adx"] = 15.0 + (quality * 10.0)
        out["kst"] = quality.astype(float) * 2.0
        out["kst_signal"] = 0.0
        out["mfi"] = 50.0 + (quality * 15.0)
        return out

    def fake_component_signals(df, **kwargs):
        del kwargs
        idx = pd.Series(range(len(df)), index=df.index)
        trade_id = idx // 3
        is_candidate = idx % 3 == 0
        direction = pd.Series(0, index=df.index, dtype="int8")
        direction.loc[is_candidate & ((trade_id % 2) == 0)] = 1
        direction.loc[is_candidate & ((trade_id % 2) == 1)] = -1
        return {
            "ema": direction.copy(),
            "rsi": direction.copy(),
            "macd": direction.copy(),
            "dmi": direction.copy(),
            "kst": direction.copy(),
            "mfi": direction.copy(),
        }

    monkeypatch.setattr("src.experiment.add_multi_indicator_columns", fake_add_multi_indicator_columns)
    monkeypatch.setattr("src.experiment.component_signals", fake_component_signals)

    base_spec = thesis_run_spec()
    spec = replace(
        base_spec,
        results_dir=str(out_dir),
        experiment=replace(
            base_spec.experiment,
            mt5=replace(base_spec.experiment.mt5, terminal_path=r"C:\Program Files\MetaTrader 5\terminal64.exe"),
            tree=replace(
                base_spec.experiment.tree,
                max_depth_grid=(1, 2),
                min_samples_leaf_grid=(1, 5),
                time_series_splits=3,
                confidence_bin_count=5,
                confidence_binning_mode="quantile",
            ),
        ),
    )

    result = run_experiment(spec, mt5_module=PatternMT5())

    metrics = pd.read_csv(result["metrics_path"])
    assert {"single_ema", "single_kst", "multi_indicator_vote", "buy_and_hold"}.issubset(set(metrics["strategy"]))
    assert "tree_confidence_range_1" in set(metrics["strategy"])
    assert "transaction_events" in metrics.columns
    assert Path(result["confidence_scores_path"]).exists()
    assert Path(result["tree_gate_path"]).exists()

    confidence = pd.read_csv(result["confidence_scores_path"])
    assert {"confidence_score", "confidence_range", "signal_direction", "label"}.issubset(confidence.columns)
    grouped = confidence.groupby("confidence_range")["label"].mean().sort_index()
    assert grouped.iloc[-1] >= grouped.iloc[0]

    rates = pd.read_csv(result["rates_path"])
    assert {"spread", "tick_volume", "real_volume"}.issubset(rates.columns)

    with Path(result["config_path"]).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    assert cfg["market_data"]["symbol"] == "US500"
    assert cfg["market_data"]["timeframe"] == "H1"
    assert cfg["market_data_timezone"] == "UTC"
    assert cfg["tree_training"]["enabled"] is True
    assert cfg["tree_training"]["validation_metric"] == "negative_brier_score"
    assert cfg["tree_training"]["min_training_samples"] == 60
    assert cfg["backtest"]["trade_size_pct"] == 0.01

    gate_text = Path(result["tree_gate_path"]).read_text(encoding="utf-8")
    assert "EvaluateDecisionTreeProbability" in gate_text
    assert "signal_direction" in gate_text
    assert "ema_fast" not in gate_text


def test_write_config_template_matches_thesis_defaults(tmp_path: Path) -> None:
    path = write_config_template(tmp_path / "template.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["results_dir"] == "results_thesis_us500_h1"
    assert data["experiment"]["market_data"]["symbol"] == "US500"
    assert data["experiment"]["market_data"]["timeframe"] == "H1"
    assert data["experiment"]["tree"]["enabled"] is True
    assert data["experiment"]["tree"]["min_training_samples"] == 60
    assert data["experiment"]["market_data"]["utc_from"].endswith("Z")
    assert data["experiment"]["market_data"]["utc_to"].endswith("Z")
    assert data["experiment"]["tree"]["confidence_bin_count"] == 5
    assert data["experiment"]["tree"]["confidence_binning_mode"] == "quantile"


def test_load_run_spec_reads_mt5_credentials_from_env(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "market_data": {"symbol": "US500"},
                    "mt5": {"login": None, "password": None, "server": None},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "secret-password")
    monkeypatch.setenv("MT5_SERVER", "Broker-Demo")

    spec = load_run_spec(cfg_path)

    assert spec.experiment.mt5.login == 123456
    assert spec.experiment.mt5.password == "secret-password"
    assert spec.experiment.mt5.server == "Broker-Demo"


def test_annualization_factor_matches_timeframe() -> None:
    assert annualization_factor("D1") == 252
    assert annualization_factor("H1") == 6048
    assert annualization_factor("M30") == 12096
