from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any

import pandas as pd


_TIMEFRAME_ATTRS: dict[str, str] = {
    "M1": "TIMEFRAME_M1",
    "M2": "TIMEFRAME_M2",
    "M3": "TIMEFRAME_M3",
    "M4": "TIMEFRAME_M4",
    "M5": "TIMEFRAME_M5",
    "M6": "TIMEFRAME_M6",
    "M10": "TIMEFRAME_M10",
    "M12": "TIMEFRAME_M12",
    "M15": "TIMEFRAME_M15",
    "M20": "TIMEFRAME_M20",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H2": "TIMEFRAME_H2",
    "H3": "TIMEFRAME_H3",
    "H4": "TIMEFRAME_H4",
    "H6": "TIMEFRAME_H6",
    "H8": "TIMEFRAME_H8",
    "H12": "TIMEFRAME_H12",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


class MT5Client:
    """Small wrapper around MetaTrader5 with deterministic dataframe output."""

    def __init__(
        self,
        terminal_path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout_ms: int = 60_000,
        portable: bool = False,
        mt5_module: Any | None = None,
    ) -> None:
        self._mt5 = mt5_module or self._load_mt5_module()
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.timeout_ms = int(timeout_ms)
        self.portable = bool(portable)
        self._connected = False

    @staticmethod
    def _load_mt5_module() -> Any:
        try:
            return importlib.import_module("MetaTrader5")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MetaTrader5 module is not available. Install it in the Python environment used to run this code."
            ) from exc

    def connect(self) -> None:
        kwargs: dict[str, Any] = {"timeout": self.timeout_ms}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if self.login is not None:
            try:
                kwargs["login"] = int(self.login)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "MetaTrader5 login must be the numeric trading account ID. "
                    f"Received login={self.login!r}. If you want to use the terminal's current logged-in session, "
                    "set login/password/server to null in the config."
                ) from exc
        if self.password:
            kwargs["password"] = self.password
        if self.server:
            kwargs["server"] = self.server
        if self.portable:
            kwargs["portable"] = True

        ok = self._mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"MetaTrader5 initialize failed: {self._mt5.last_error()}")
        self._connected = True

    def shutdown(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False

    def __enter__(self) -> MT5Client:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()

    def _resolve_timeframe(self, timeframe: str) -> int:
        key = timeframe.strip().upper()
        attr = _TIMEFRAME_ATTRS.get(key)
        if attr is None:
            supported = ", ".join(sorted(_TIMEFRAME_ATTRS.keys()))
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported values: {supported}")

        if not hasattr(self._mt5, attr):
            raise RuntimeError(f"MetaTrader5 does not expose {attr}")
        return int(getattr(self._mt5, attr))

    def _ensure_symbol(self, symbol: str) -> None:
        if not self._mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select failed for '{symbol}': {self._mt5.last_error()}")

    @staticmethod
    def _normalize_rates(rates: Any) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            raise RuntimeError("MetaTrader5 returned no rates for this request")

        df = pd.DataFrame(rates)
        required = {"time", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"MetaTrader5 rates missing required columns: {sorted(missing)}")

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        tick_volume = pd.to_numeric(df.get("tick_volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        spread = pd.to_numeric(df.get("spread", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        real_volume = pd.to_numeric(df.get("real_volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        if "real_volume" in df.columns:
            df["volume"] = real_volume.where(real_volume > 0, tick_volume)
        else:
            df["volume"] = tick_volume

        out = df[["time", "open", "high", "low", "close", "volume"]].copy()
        out["tick_volume"] = tick_volume
        out["spread"] = spread
        out["real_volume"] = real_volume
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
        out["tick_volume"] = pd.to_numeric(out["tick_volume"], errors="coerce").fillna(0)
        out["spread"] = pd.to_numeric(out["spread"], errors="coerce").fillna(0)
        out["real_volume"] = pd.to_numeric(out["real_volume"], errors="coerce").fillna(0)
        out = out.drop_duplicates(subset=["time"]).sort_values("time").set_index("time")
        out["ret"] = out["close"].pct_change().fillna(0.0)
        return out

    def fetch_rates(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
        utc_from: datetime | None = None,
        utc_to: datetime | None = None,
    ) -> pd.DataFrame:
        if not self._connected:
            raise RuntimeError("MetaTrader5 client is not connected")

        self._ensure_symbol(symbol)
        tf = self._resolve_timeframe(timeframe)

        if utc_from is not None and utc_to is not None:
            rates = self._mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
        else:
            rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, int(bars))

        return self._normalize_rates(rates)
