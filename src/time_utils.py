from __future__ import annotations

from datetime import datetime

import pandas as pd


def parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()
