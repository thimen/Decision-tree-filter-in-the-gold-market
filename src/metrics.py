from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    return float(drawdown.min())


def compute_performance(
    bt: BacktestResult,
    trade_log: pd.DataFrame | None = None,
    annualization: int = 252,
) -> dict[str, float]:
    equity = bt.equity.ffill().fillna(1.0)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    closed_trades = pd.DataFrame() if trade_log is None else trade_log.loc[trade_log.get("status", pd.Series(dtype="object")) == "closed"].copy()

    n = len(returns)
    if n == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "transaction_events": 0.0,
            "win_rate": 0.0,
            "average_trade_return": 0.0,
            "profit_factor": 0.0,
        }

    initial_equity = float(equity.iloc[0]) if len(equity) else 1.0
    final_equity = float(equity.iloc[-1]) if len(equity) else initial_equity
    growth = final_equity / initial_equity if initial_equity != 0 else 0.0
    total_return = growth - 1.0
    years = max(n / float(annualization), 1e-9)
    cagr = float(growth ** (1.0 / years) - 1.0) if growth > 0 else -1.0

    std = returns.std(ddof=0)
    volatility = float(std * np.sqrt(annualization))
    sharpe = float((returns.mean() / std) * np.sqrt(annualization)) if std > 0 else 0.0

    win_rate = 0.0
    average_trade_return = 0.0
    profit_factor = 0.0
    if not closed_trades.empty and "gross_return_pct" in closed_trades.columns:
        gross = closed_trades["gross_return_pct"].astype(float)
        wins = gross[gross > 0.0]
        losses = gross[gross < 0.0]
        win_rate = float((gross > 0.0).mean())
        average_trade_return = float(gross.mean())
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)

    if trade_log is not None and not trade_log.empty:
        trade_count = int(len(closed_trades))
    else:
        trade_count = int((bt.trades > 0).sum())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "trades": int(trade_count),
        "transaction_events": float(bt.trades.sum()),
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "profit_factor": profit_factor,
    }
