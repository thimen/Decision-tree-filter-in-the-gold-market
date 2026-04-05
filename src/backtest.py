from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    net_returns: pd.Series
    position: pd.Series
    trades: pd.Series
    equity: pd.Series


@dataclass(frozen=True)
class ExecutionTrace:
    result: BacktestResult
    events: pd.DataFrame
    trade_log: pd.DataFrame


@dataclass
class OpenTrade:
    direction: int
    entry_price: float
    entry_time: Any
    entry_bar: int
    mark_price: float
    mark_value: float


@dataclass
class TraceBuffers:
    net_returns: pd.Series
    position: pd.Series
    trades: pd.Series
    entry_event: pd.Series
    exit_event: pd.Series
    entry_count: pd.Series
    exit_count: pd.Series
    entry_fill_price: pd.Series
    exit_fill_price: pd.Series
    entry_direction: pd.Series
    exit_direction: pd.Series
    exit_reason: pd.Series
    trade_rows: list[dict[str, Any]]


def directional_return(entry_price: float, exit_price: float, direction: int) -> float:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return float(direction) * ((float(exit_price) / float(entry_price)) - 1.0)


def resolve_long_exit(
    entry_price: float,
    high: float,
    low: float,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
) -> tuple[str, float] | None:
    take_profit_price = entry_price * (1.0 + take_profit_pct) if take_profit_pct is not None else None
    stop_loss_price = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct is not None else None
    hit_tp = take_profit_price is not None and high >= take_profit_price
    hit_sl = stop_loss_price is not None and low <= stop_loss_price

    if hit_sl:
        return ("stop_loss", float(stop_loss_price))
    if hit_tp:
        return ("take_profit", float(take_profit_price))
    return None


def resolve_short_exit(
    entry_price: float,
    high: float,
    low: float,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
) -> tuple[str, float] | None:
    take_profit_price = entry_price * (1.0 - take_profit_pct) if take_profit_pct is not None else None
    stop_loss_price = entry_price * (1.0 + stop_loss_pct) if stop_loss_pct is not None else None
    hit_tp = take_profit_price is not None and low <= take_profit_price
    hit_sl = stop_loss_price is not None and high >= stop_loss_price

    if hit_sl:
        return ("stop_loss", float(stop_loss_price))
    if hit_tp:
        return ("take_profit", float(take_profit_price))
    return None


def resolve_exit(
    direction: int,
    entry_price: float,
    high: float,
    low: float,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
) -> tuple[str, float] | None:
    if direction == 1:
        return resolve_long_exit(
            entry_price=entry_price,
            high=high,
            low=low,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
    if direction == -1:
        return resolve_short_exit(
            entry_price=entry_price,
            high=high,
            low=low,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
    return None


def _validate_exit_thresholds(
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
) -> None:
    if take_profit_pct is not None and take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be > 0 when set")
    if stop_loss_pct is not None and stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be > 0 when set")


def entry_event_mask(
    signal: pd.Series,
    long_only: bool = False,
) -> pd.Series:
    cmd = signal.fillna(0).astype("int8")
    if long_only:
        cmd = cmd.clip(lower=0, upper=1)

    if len(cmd) == 0:
        return pd.Series(dtype="bool", index=cmd.index)

    out = pd.Series(False, index=cmd.index)
    prev = cmd.shift(1, fill_value=0).astype("int8")
    out = (cmd != 0) & (cmd != prev)
    return out


def _finalize_trade(
    trade_rows: list[dict[str, Any]],
    entry_time: Any,
    entry_price: float,
    exit_time: Any,
    exit_price: float,
    exit_reason: str,
    bars_held: int,
    direction: int,
) -> None:
    trade_rows.append(
        {
            "side": "long" if direction == 1 else "short",
            "direction": int(direction),
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "gross_return_pct": directional_return(entry_price, exit_price, direction),
            "status": "closed",
        }
    )


def _initialize_trace_buffers(index: pd.Index) -> TraceBuffers:
    return TraceBuffers(
        net_returns=pd.Series(0.0, index=index),
        position=pd.Series(0.0, index=index),
        trades=pd.Series(0.0, index=index),
        entry_event=pd.Series(False, index=index),
        exit_event=pd.Series(False, index=index),
        entry_count=pd.Series(0, index=index, dtype="int64"),
        exit_count=pd.Series(0, index=index, dtype="int64"),
        entry_fill_price=pd.Series(float("nan"), index=index),
        exit_fill_price=pd.Series(float("nan"), index=index),
        entry_direction=pd.Series(0, index=index, dtype="int8"),
        exit_direction=pd.Series(0, index=index, dtype="int8"),
        exit_reason=pd.Series("", index=index, dtype="object"),
        trade_rows=[],
    )


def _register_entry(
    *,
    buffers: TraceBuffers,
    open_trades: list[OpenTrade],
    command: int,
    fill_price: float,
    fill_time: Any,
    entry_bar: int,
    bar_index: int,
    portfolio_equity: float,
    trade_size_pct: float,
    cost: float,
) -> float:
    allocation = portfolio_equity * float(trade_size_pct)
    if command not in (-1, 1) or allocation <= 0.0:
        return portfolio_equity

    entry_cost = allocation * cost
    portfolio_equity -= entry_cost
    buffers.net_returns.iloc[bar_index] -= entry_cost
    buffers.trades.iloc[bar_index] += 1.0
    open_trades.append(
        OpenTrade(
            direction=command,
            entry_price=fill_price,
            entry_time=fill_time,
            entry_bar=entry_bar,
            mark_price=fill_price,
            mark_value=allocation,
        )
    )
    buffers.entry_event.iloc[entry_bar] = True
    buffers.entry_count.iloc[entry_bar] += 1
    buffers.entry_fill_price.iloc[entry_bar] = fill_price
    buffers.entry_direction.iloc[entry_bar] = command
    return portfolio_equity


def _close_trade(
    *,
    buffers: TraceBuffers,
    trade: OpenTrade,
    exit_price: float,
    exit_reason: str,
    bar_index: int,
    exit_time: Any,
    cost: float,
    portfolio_equity: float,
) -> float:
    step_return = directional_return(trade.mark_price, exit_price, trade.direction)
    previous_value = trade.mark_value
    next_value = previous_value * (1.0 + step_return)
    pnl_delta = next_value - previous_value
    portfolio_equity += pnl_delta
    buffers.net_returns.iloc[bar_index] += pnl_delta
    buffers.exit_event.iloc[bar_index] = True
    buffers.exit_count.iloc[bar_index] += 1
    if pd.isna(buffers.exit_fill_price.iloc[bar_index]):
        buffers.exit_fill_price.iloc[bar_index] = exit_price
        buffers.exit_direction.iloc[bar_index] = trade.direction
        buffers.exit_reason.iloc[bar_index] = exit_reason

    _finalize_trade(
        trade_rows=buffers.trade_rows,
        entry_time=trade.entry_time,
        entry_price=trade.entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        bars_held=bar_index - trade.entry_bar,
        direction=trade.direction,
    )

    exit_cost = next_value * cost
    portfolio_equity -= exit_cost
    buffers.trades.iloc[bar_index] += 1.0
    buffers.net_returns.iloc[bar_index] -= exit_cost
    return portfolio_equity


def _mark_trade_to_market(
    *,
    buffers: TraceBuffers,
    trade: OpenTrade,
    bar_close: float,
    bar_index: int,
    portfolio_equity: float,
) -> tuple[float, OpenTrade]:
    step_return = directional_return(trade.mark_price, bar_close, trade.direction)
    previous_value = trade.mark_value
    next_value = previous_value * (1.0 + step_return)
    pnl_delta = next_value - previous_value
    portfolio_equity += pnl_delta
    buffers.net_returns.iloc[bar_index] += pnl_delta
    trade.mark_price = bar_close
    trade.mark_value = next_value
    return portfolio_equity, trade


def _append_open_trade_rows(
    trade_rows: list[dict[str, Any]],
    open_trades: list[OpenTrade],
    df: pd.DataFrame,
) -> None:
    for trade in open_trades:
        trade_rows.append(
            {
                "side": "long" if trade.direction == 1 else "short",
                "direction": int(trade.direction),
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_time": None,
                "exit_price": None,
                "exit_reason": "open",
                "bars_held": len(df) - 1 - trade.entry_bar,
                "gross_return_pct": directional_return(trade.entry_price, float(df["close"].iloc[-1]), trade.direction),
                "status": "open",
            }
        )


def _build_events_frame(buffers: TraceBuffers) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_event": buffers.entry_event,
            "entry_count": buffers.entry_count,
            "entry_fill_price": buffers.entry_fill_price,
            "entry_direction": buffers.entry_direction,
            "exit_event": buffers.exit_event,
            "exit_count": buffers.exit_count,
            "exit_fill_price": buffers.exit_fill_price,
            "exit_direction": buffers.exit_direction,
            "exit_reason": buffers.exit_reason,
        }
    )


def simulate_with_trace(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
    initial_cash: float = 1.0,
    trade_size_pct: float = 0.01,
    long_only: bool = False,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> ExecutionTrace:
    required = {"close", "high", "low", "ret"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe must contain {sorted(required)}")

    _validate_exit_thresholds(take_profit_pct, stop_loss_pct)
    if float(initial_cash) <= 0:
        raise ValueError("initial_cash must be > 0")
    if float(trade_size_pct) <= 0:
        raise ValueError("trade_size_pct must be > 0")

    aligned_signal = signal.reindex(df.index).fillna(0).astype("int8")
    if long_only:
        aligned_signal = aligned_signal.clip(lower=0, upper=1)
    aligned_entries = aligned_signal.where(entry_event_mask(aligned_signal, long_only=long_only), 0).astype("int8")
    returns = df["ret"].fillna(0.0)
    buffers = _initialize_trace_buffers(df.index)

    cost = float(cost_bps) / 10_000.0
    portfolio_equity = float(initial_cash)
    open_trades: list[OpenTrade] = []

    for i in range(1, len(df)):
        command = int(aligned_entries.iloc[i - 1])
        fill_price = float(df["close"].iloc[i - 1])
        fill_time = df.index[i - 1]
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])
        bar_close = float(df["close"].iloc[i])

        portfolio_equity = _register_entry(
            buffers=buffers,
            open_trades=open_trades,
            command=command,
            fill_price=fill_price,
            fill_time=fill_time,
            entry_bar=i - 1,
            bar_index=i,
            portfolio_equity=portfolio_equity,
            trade_size_pct=trade_size_pct,
            cost=cost,
        )

        active_trades: list[OpenTrade] = []
        for trade in open_trades:
            exit_fill = resolve_exit(
                direction=trade.direction,
                entry_price=trade.entry_price,
                high=bar_high,
                low=bar_low,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
            )
            if exit_fill is not None:
                reason, price = exit_fill
                portfolio_equity = _close_trade(
                    buffers=buffers,
                    trade=trade,
                    exit_price=price,
                    exit_reason=reason,
                    bar_index=i,
                    exit_time=df.index[i],
                    cost=cost,
                    portfolio_equity=portfolio_equity,
                )
                continue

            portfolio_equity, trade = _mark_trade_to_market(
                buffers=buffers,
                trade=trade,
                bar_close=bar_close,
                bar_index=i,
                portfolio_equity=portfolio_equity,
            )
            active_trades.append(trade)

        open_trades = active_trades
        active_exposure = sum(trade.mark_value for trade in open_trades)
        buffers.position.iloc[i] = float(active_exposure / portfolio_equity) if portfolio_equity > 0 else 0.0

    _append_open_trade_rows(buffers.trade_rows, open_trades, df)

    equity = float(initial_cash) + buffers.net_returns.cumsum()
    result = BacktestResult(
        returns=returns,
        net_returns=buffers.net_returns,
        position=buffers.position,
        trades=buffers.trades,
        equity=equity,
    )
    events = _build_events_frame(buffers)
    trade_log = pd.DataFrame(buffers.trade_rows)
    return ExecutionTrace(result=result, events=events, trade_log=trade_log)


def simulate(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
    initial_cash: float = 1.0,
    trade_size_pct: float = 0.01,
    long_only: bool = False,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> BacktestResult:
    return simulate_with_trace(
        df,
        signal=signal,
        cost_bps=cost_bps,
        initial_cash=initial_cash,
        trade_size_pct=trade_size_pct,
        long_only=long_only,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    ).result
