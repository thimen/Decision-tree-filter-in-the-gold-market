from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_equity_curves(curves: dict[str, pd.Series], output_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for name, curve in curves.items():
        plt.plot(curve.index, curve.values, label=name)
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(equity: pd.Series, output_path: str | Path, title: str = "Drawdown") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown.values, 0.0, alpha=0.4)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trade_markers(
    close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    trade_log: pd.DataFrame,
    output_path: str | Path,
    title: str = "Trade Markers",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(close.index, close.values, label="Close", color="black", linewidth=1.0)
    plt.plot(ema_fast.index, ema_fast.values, label="EMA Fast", color="#1f77b4", linewidth=1.0, alpha=0.9)
    plt.plot(ema_slow.index, ema_slow.values, label="EMA Slow", color="#ff7f0e", linewidth=1.0, alpha=0.9)

    if not trade_log.empty:
        closed = trade_log[trade_log["status"] == "closed"].copy()
        open_trades = trade_log[trade_log["status"] == "open"].copy()

        plt.scatter(
            pd.to_datetime(trade_log["entry_time"]),
            trade_log["entry_price"],
            marker="^",
            color="#2ca02c",
            s=36,
            label="Buy",
            zorder=3,
        )

        if not closed.empty:
            tp = closed[closed["exit_reason"] == "take_profit"]
            sl = closed[closed["exit_reason"] == "stop_loss"]
            if not tp.empty:
                plt.scatter(pd.to_datetime(tp["exit_time"]), tp["exit_price"], marker="v", color="#1f77b4", s=36, label="Take Profit", zorder=3)
            if not sl.empty:
                plt.scatter(pd.to_datetime(sl["exit_time"]), sl["exit_price"], marker="v", color="#d62728", s=36, label="Stop Loss", zorder=3)

        if not open_trades.empty:
            plt.scatter(
                pd.to_datetime(open_trades["entry_time"]),
                open_trades["entry_price"],
                marker="o",
                facecolors="none",
                edgecolors="#2ca02c",
                s=60,
                label="Open Trade",
                zorder=3,
            )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
