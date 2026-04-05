Copy [DecisionTreeMultiIndicatorEA.mq5](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/mql5/Experts/DecisionTreeMultiIndicatorEA.mq5) into MT5 `MQL5/Experts/`.

Copy [TreeGate.mqh](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/mql5/Include/TreeGate.mqh) into MT5 `MQL5/Include/`.

The files are aligned with the final thesis run:

- instrument: `XAUUSD`
- timeframe: `H1`
- requested data window: `2022-01-01T00:00:00Z` to `2024-12-31T23:00:00Z`
- strategy: multi-indicator vote using `EMA`, `RSI`, `MACD`, `DMI`, `KST`, and `MFI`
- exit logic: `take_profit_pct=0.06`, `stop_loss_pct=0.03`
- tree gate: exported from [results_xauusd_h1_3y/tree_gate_generated.mqh](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/results_xauusd_h1_3y/tree_gate_generated.mqh)

Compile in MetaEditor with `F7`, then run the EA in MT5 Strategy Tester on `XAUUSD` `H1`.

Use the same parameters that produced the Python thesis results. The main thesis config is [run_config.xauusd_h1.json](/Users/thimthor/Desktop/skola/Examensarbete/Kod1/configs/run_config.xauusd_h1.json).

Important EA inputs:

- `InpUseTreeFilter`
- `InpConfidenceMin`
- `InpConfidenceMax`
- `InpTakeProfitPct`
- `InpStopLossPct`
- indicator periods and thresholds matching the Python run

Suggested Strategy Tester validation workflow:

1. Run once with `InpUseTreeFilter=false` to compare against the unfiltered multi-indicator baseline.
2. Run once with `InpUseTreeFilter=true`, `InpConfidenceMin=0.3919597990`, and `InpConfidenceMax=0.3919597990` to validate the strongest thesis band (`Range 4`).
3. Optionally run all five thesis confidence bands:
   - `Range 1`: `0.1800000000` to `0.1913043478`
   - `Range 2`: `0.2631578947` to `0.2631578947`
   - `Range 3`: `0.3620689655` to `0.3863636364`
   - `Range 4`: `0.3919597990` to `0.3919597990`
   - `Range 5`: `0.4347826087` to `0.5161290323`
4. Keep the tester mode, spread assumptions, and symbol/timeframe fixed across runs.

MT5-native outputs:

- `Results`: trade-by-trade log
- `Graph`: equity and balance curves
- `Report`: profitability, drawdown, win rate, profit factor, Sharpe, and related metrics
- `Journal`: runtime diagnostics
