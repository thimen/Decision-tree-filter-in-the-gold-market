from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.mt5_client import MT5Client
from src.run.configuration import load_run_spec
from src.time_utils import parse_utc


def check_connection(config_path: str | Path) -> dict[str, Any]:
    spec = load_run_spec(config_path)
    cfg = spec.experiment
    with MT5Client(
        terminal_path=cfg.mt5.terminal_path,
        login=cfg.mt5.login,
        password=cfg.mt5.password,
        server=cfg.mt5.server,
        timeout_ms=cfg.mt5.timeout_ms,
        portable=cfg.mt5.portable,
    ) as client:
        rates = client.fetch_rates(
            symbol=cfg.market_data.symbol,
            timeframe=cfg.market_data.timeframe,
            bars=5,
            utc_from=parse_utc(cfg.market_data.utc_from),
            utc_to=parse_utc(cfg.market_data.utc_to),
        )

    return {
        "ok": True,
        "terminal_path": cfg.mt5.terminal_path,
        "symbol": cfg.market_data.symbol,
        "timeframe": cfg.market_data.timeframe,
        "rows": int(len(rates)),
        "first_time": str(rates.index[0]) if len(rates) else "",
        "last_time": str(rates.index[-1]) if len(rates) else "",
        "columns": list(rates.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MT5 auth and data access for a run config")
    parser.add_argument("--config", required=True, help="Path to JSON run configuration file")
    args = parser.parse_args()
    print(json.dumps(check_connection(args.config), indent=2))


if __name__ == "__main__":
    main()
