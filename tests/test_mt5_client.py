from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.mt5_client import MT5Client


class FakeMT5:
    TIMEFRAME_D1 = 1440
    TIMEFRAME_H1 = 60

    def __init__(self) -> None:
        self.shutdown_called = False
        self.range_calls = 0
        self.pos_calls = 0

    def initialize(self, **kwargs):
        del kwargs
        return True

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (1, "error")

    def symbol_select(self, symbol, enabled):
        return symbol == "US500" and enabled

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, bars):
        del symbol, timeframe, start_pos, bars
        self.pos_calls += 1
        return [
            {
                "time": 1704067200,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "tick_volume": 10,
                "spread": 12,
                "real_volume": 0,
            },
            {
                "time": 1704070800,
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "tick_volume": 12,
                "spread": 15,
                "real_volume": 20,
            },
        ]

    def copy_rates_range(self, symbol, timeframe, utc_from, utc_to):
        del symbol, timeframe, utc_from, utc_to
        self.range_calls += 1
        return [
            {
                "time": 1704067200,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "tick_volume": 10,
                "spread": 11,
                "real_volume": 5,
            }
        ]


def test_fetch_rates_from_pos_preserves_thesis_fields() -> None:
    mt5 = FakeMT5()
    with MT5Client(terminal_path="C:/Program Files/MetaTrader 5/terminal64.exe", mt5_module=mt5) as client:
        df = client.fetch_rates(symbol="US500", timeframe="D1", bars=2)

    assert mt5.pos_calls == 1
    assert mt5.shutdown_called
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "tick_volume", "spread", "real_volume", "ret"]
    assert str(df.index.tz) == "UTC"
    assert df.iloc[0]["volume"] == 10
    assert df.iloc[1]["volume"] == 20
    assert df.iloc[0]["spread"] == 12
    assert df.iloc[1]["real_volume"] == 20


def test_fetch_rates_range_uses_copy_rates_range() -> None:
    mt5 = FakeMT5()
    with MT5Client(terminal_path="C:/Program Files/MetaTrader 5/terminal64.exe", mt5_module=mt5) as client:
        df = client.fetch_rates(
            symbol="US500",
            timeframe="H1",
            bars=100,
            utc_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            utc_to=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

    assert not df.empty
    assert mt5.range_calls == 1
    assert float(df.iloc[0]["spread"]) == 11


def test_unsupported_timeframe_raises() -> None:
    mt5 = FakeMT5()
    with MT5Client(terminal_path="C:/Program Files/MetaTrader 5/terminal64.exe", mt5_module=mt5) as client:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            client.fetch_rates(symbol="US500", timeframe="M99", bars=10)


def test_symbol_select_failure_raises() -> None:
    mt5 = FakeMT5()
    with MT5Client(terminal_path="C:/Program Files/MetaTrader 5/terminal64.exe", mt5_module=mt5) as client:
        with pytest.raises(RuntimeError, match="symbol_select failed"):
            client.fetch_rates(symbol="INVALID", timeframe="D1", bars=10)
