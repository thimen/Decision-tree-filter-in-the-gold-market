#property strict

#include <Trade/Trade.mqh>
#include <TreeGate.mqh>

enum OscillatorVoteMode
{
   OSCILLATOR_TREND = 0,
   OSCILLATOR_MEAN_REVERSION = 1
};

input group "Execution"
input double InpLots = 0.10;
input int InpDeviationPoints = 20;
input ulong InpMagicNumber = 20260326;
input bool InpUseTreeFilter = true;
input double InpConfidenceMin = 0.0;
input double InpConfidenceMax = 1.0;
input double InpTakeProfitPct = 0.06;
input double InpStopLossPct = 0.03;

input group "Strategy"
input int InpEmaFast = 19;
input int InpEmaSlow = 27;
input int InpVoteK = 3;
input OscillatorVoteMode InpOscillatorMode = OSCILLATOR_TREND;
input int InpRsiPeriod = 21;
input double InpRsiBuy = 45.0;
input double InpRsiSell = 55.0;
input int InpMacdFast = 21;
input int InpMacdSlow = 29;
input int InpMacdSignal = 9;
input int InpDmiPeriod = 14;
input int InpMfiPeriod = 14;
input double InpMfiBuy = 45.0;
input double InpMfiSell = 55.0;
input bool InpIncludeEmaVote = true;
input int InpKstRoc1 = 10;
input int InpKstRoc2 = 15;
input int InpKstRoc3 = 20;
input int InpKstRoc4 = 30;
input int InpKstSma1 = 10;
input int InpKstSma2 = 10;
input int InpKstSma3 = 10;
input int InpKstSma4 = 15;
input int InpKstSignal = 9;

CTrade trade;
datetime g_last_bar_time = 0;
int g_prev_signal_direction = 0;
int g_handle_ema_fast = INVALID_HANDLE;
int g_handle_ema_slow = INVALID_HANDLE;
int g_handle_rsi = INVALID_HANDLE;
int g_handle_macd = INVALID_HANDLE;
int g_handle_adx = INVALID_HANDLE;
int g_handle_mfi = INVALID_HANDLE;


bool CopyValue(const int handle, const int buffer, const int shift, double &value)
{
   double data[];
   ArrayResize(data, 1);
   int copied = CopyBuffer(handle, buffer, shift, 1, data);
   if(copied != 1)
      return false;
   value = data[0];
   return (value != EMPTY_VALUE);
}


double NormalizePrice(const double price)
{
   return NormalizeDouble(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
}


bool IsNewBar()
{
   datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_time <= 0)
      return false;

   if(g_last_bar_time == 0)
   {
      g_last_bar_time = current_bar_time;
      return false;
   }

   if(current_bar_time == g_last_bar_time)
      return false;

   g_last_bar_time = current_bar_time;
   return true;
}


int VoteFromSign(const double lhs, const double rhs)
{
   if(lhs > rhs)
      return 1;
   if(lhs < rhs)
      return -1;
   return 0;
}


int CountVotes(const int v1, const int v2, const int v3, const int v4, const int v5, const bool include_ema, const int ema_vote, const int target)
{
   int count = 0;
   if(v1 == target) count++;
   if(v2 == target) count++;
   if(v3 == target) count++;
   if(v4 == target) count++;
   if(v5 == target) count++;
   if(include_ema && ema_vote == target) count++;
   return count;
}


bool ComputeSmoothedROC(const double &closes[], const int shift, const int roc_period, const int sma_period, double &value)
{
   double sum = 0.0;
   for(int offset = 0; offset < sma_period; ++offset)
   {
      int current_index = shift + offset;
      int base_index = current_index + roc_period;
      if(base_index >= ArraySize(closes))
         return false;

      double base = closes[base_index];
      if(base == 0.0)
         return false;
      sum += ((closes[current_index] - base) / base) * 100.0;
   }

   value = sum / (double)sma_period;
   return true;
}


bool ComputeKSTRaw(const double &closes[], const int shift, double &kst_value)
{
   double rcma1 = 0.0;
   double rcma2 = 0.0;
   double rcma3 = 0.0;
   double rcma4 = 0.0;

   if(!ComputeSmoothedROC(closes, shift, InpKstRoc1, InpKstSma1, rcma1)) return false;
   if(!ComputeSmoothedROC(closes, shift, InpKstRoc2, InpKstSma2, rcma2)) return false;
   if(!ComputeSmoothedROC(closes, shift, InpKstRoc3, InpKstSma3, rcma3)) return false;
   if(!ComputeSmoothedROC(closes, shift, InpKstRoc4, InpKstSma4, rcma4)) return false;

   kst_value = rcma1 + (2.0 * rcma2) + (3.0 * rcma3) + (4.0 * rcma4);
   return true;
}


bool ComputeKSTForShift(const double &closes[], const int shift, double &kst_value, double &kst_signal_value)
{
   double signal_sum = 0.0;
   for(int offset = 0; offset < InpKstSignal; ++offset)
   {
      double raw_value = 0.0;
      if(!ComputeKSTRaw(closes, shift + offset, raw_value))
         return false;
      if(offset == 0)
         kst_value = raw_value;
      signal_sum += raw_value;
   }

   kst_signal_value = signal_sum / (double)InpKstSignal;
   return true;
}


bool ComputeKSTState(double &kst_1, double &kst_signal_1, double &kst_2, double &kst_signal_2)
{
   int bars_needed = InpKstRoc4 + InpKstSma4 + InpKstSignal + 5;
   double closes[];
   ArrayResize(closes, bars_needed);
   ArraySetAsSeries(closes, true);

   int copied = CopyClose(_Symbol, PERIOD_CURRENT, 0, bars_needed, closes);
   if(copied < bars_needed)
      return false;

   return ComputeKSTForShift(closes, 1, kst_1, kst_signal_1) && ComputeKSTForShift(closes, 2, kst_2, kst_signal_2);
}


void OpenManagedPosition(const int direction)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
   {
      Print("Cannot open trade: invalid market prices");
      return;
   }

   bool ok = false;
   if(direction == 1)
   {
      double stop_loss = NormalizePrice(ask * (1.0 - InpStopLossPct));
      double take_profit = NormalizePrice(ask * (1.0 + InpTakeProfitPct));
      ok = trade.Buy(InpLots, _Symbol, 0.0, stop_loss, take_profit, "ThesisDecisionTreeEA");
   }
   else if(direction == -1)
   {
      double stop_loss = NormalizePrice(bid * (1.0 + InpStopLossPct));
      double take_profit = NormalizePrice(bid * (1.0 - InpTakeProfitPct));
      ok = trade.Sell(InpLots, _Symbol, 0.0, stop_loss, take_profit, "ThesisDecisionTreeEA");
   }

   if(!ok)
      Print("Order failed: ", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
}


void EvaluateSignalOnClosedBar()
{
   double ema_fast_1 = 0.0;
   double ema_slow_1 = 0.0;
   double rsi_1 = 0.0;
   double macd_1 = 0.0;
   double macd_signal_1 = 0.0;
   double macd_2 = 0.0;
   double macd_signal_2 = 0.0;
   double plus_di_1 = 0.0;
   double minus_di_1 = 0.0;
   double adx_1 = 0.0;
   double mfi_1 = 0.0;
   double kst_1 = 0.0;
   double kst_signal_1 = 0.0;
   double kst_2 = 0.0;
   double kst_signal_2 = 0.0;

   if(!CopyValue(g_handle_ema_fast, 0, 1, ema_fast_1) ||
      !CopyValue(g_handle_ema_slow, 0, 1, ema_slow_1) ||
      !CopyValue(g_handle_rsi, 0, 1, rsi_1) ||
      !CopyValue(g_handle_macd, 0, 1, macd_1) ||
      !CopyValue(g_handle_macd, 1, 1, macd_signal_1) ||
      !CopyValue(g_handle_macd, 0, 2, macd_2) ||
      !CopyValue(g_handle_macd, 1, 2, macd_signal_2) ||
      !CopyValue(g_handle_adx, 1, 1, plus_di_1) ||
      !CopyValue(g_handle_adx, 2, 1, minus_di_1) ||
      !CopyValue(g_handle_adx, 0, 1, adx_1) ||
      !CopyValue(g_handle_mfi, 0, 1, mfi_1) ||
      !ComputeKSTState(kst_1, kst_signal_1, kst_2, kst_signal_2))
   {
      Print("Skipping bar: indicator data not ready");
      return;
   }

   int ema_vote = VoteFromSign(ema_fast_1, ema_slow_1);
   int dmi_vote = VoteFromSign(plus_di_1, minus_di_1);

   int macd_vote = 0;
   if(macd_1 > macd_signal_1 && macd_2 <= macd_signal_2)
      macd_vote = 1;
   else if(macd_1 < macd_signal_1 && macd_2 >= macd_signal_2)
      macd_vote = -1;

   int kst_vote = 0;
   if(kst_1 > kst_signal_1 && kst_2 <= kst_signal_2)
      kst_vote = 1;
   else if(kst_1 < kst_signal_1 && kst_2 >= kst_signal_2)
      kst_vote = -1;

   int rsi_vote = 0;
   int mfi_vote = 0;
   if(InpOscillatorMode == OSCILLATOR_MEAN_REVERSION)
   {
      if(rsi_1 < InpRsiBuy) rsi_vote = 1;
      else if(rsi_1 > InpRsiSell) rsi_vote = -1;

      if(mfi_1 < InpMfiBuy) mfi_vote = 1;
      else if(mfi_1 > InpMfiSell) mfi_vote = -1;
   }
   else
   {
      if(rsi_1 > InpRsiSell) rsi_vote = 1;
      else if(rsi_1 < InpRsiBuy) rsi_vote = -1;

      if(mfi_1 > InpMfiSell) mfi_vote = 1;
      else if(mfi_1 < InpMfiBuy) mfi_vote = -1;
   }

   int buy_vote_count = CountVotes(rsi_vote, macd_vote, dmi_vote, kst_vote, mfi_vote, InpIncludeEmaVote, ema_vote, 1);
   int sell_vote_count = CountVotes(rsi_vote, macd_vote, dmi_vote, kst_vote, mfi_vote, InpIncludeEmaVote, ema_vote, -1);

   int signal_direction = 0;
   if(buy_vote_count >= InpVoteK && buy_vote_count > sell_vote_count)
      signal_direction = 1;
   else if(sell_vote_count >= InpVoteK && sell_vote_count > buy_vote_count)
      signal_direction = -1;

   bool is_entry_event = (signal_direction != 0 && signal_direction != g_prev_signal_direction);
   g_prev_signal_direction = signal_direction;

   if(!is_entry_event)
      return;

   double ema_gap = ema_fast_1 - ema_slow_1;
   double macd_gap = macd_1 - macd_signal_1;
   double di_gap = plus_di_1 - minus_di_1;
   double kst_gap = kst_1 - kst_signal_1;

   double confidence = 1.0;
   if(InpUseTreeFilter)
   {
      confidence = EvaluateDecisionTreeProbability(
         rsi_1,
         adx_1,
         mfi_1,
         ema_gap,
         macd_gap,
         di_gap,
         kst_gap,
         (double)ema_vote,
         (double)rsi_vote,
         (double)macd_vote,
         (double)dmi_vote,
         (double)kst_vote,
         (double)mfi_vote,
         (double)buy_vote_count,
         (double)sell_vote_count,
         (double)signal_direction,
         (signal_direction == 1 ? 1.0 : 0.0),
         (signal_direction == -1 ? 1.0 : 0.0)
      );
      if(confidence < InpConfidenceMin || confidence > InpConfidenceMax)
         return;
   }

   OpenManagedPosition(signal_direction);
}


int OnInit()
{
   if((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE) != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
   {
      Print("This EA requires a hedging account to match the thesis execution model");
      return INIT_PARAMETERS_INCORRECT;
   }

   if(InpVoteK <= 0)
   {
      Print("InpVoteK must be > 0");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpConfidenceMin < 0.0 || InpConfidenceMax > 1.0 || InpConfidenceMin > InpConfidenceMax)
   {
      Print("Confidence range must be inside [0, 1] and min <= max");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpTakeProfitPct <= 0.0 || InpStopLossPct <= 0.0)
   {
      Print("TakeProfitPct and StopLossPct must be > 0");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpRsiBuy >= InpRsiSell || InpMfiBuy >= InpMfiSell)
   {
      Print("Buy thresholds must be below sell thresholds");
      return INIT_PARAMETERS_INCORRECT;
   }

   g_handle_ema_fast = iMA(_Symbol, PERIOD_CURRENT, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   g_handle_ema_slow = iMA(_Symbol, PERIOD_CURRENT, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   g_handle_rsi = iRSI(_Symbol, PERIOD_CURRENT, InpRsiPeriod, PRICE_CLOSE);
   g_handle_macd = iMACD(_Symbol, PERIOD_CURRENT, InpMacdFast, InpMacdSlow, InpMacdSignal, PRICE_CLOSE);
   g_handle_adx = iADX(_Symbol, PERIOD_CURRENT, InpDmiPeriod);
   g_handle_mfi = iMFI(_Symbol, PERIOD_CURRENT, InpMfiPeriod, VOLUME_TICK);

   if(g_handle_ema_fast == INVALID_HANDLE || g_handle_ema_slow == INVALID_HANDLE || g_handle_rsi == INVALID_HANDLE ||
      g_handle_macd == INVALID_HANDLE || g_handle_adx == INVALID_HANDLE || g_handle_mfi == INVALID_HANDLE)
   {
      Print("Failed to create one or more indicator handles");
      return INIT_FAILED;
   }

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);
   return INIT_SUCCEEDED;
}


void OnDeinit(const int reason)
{
   if(g_handle_ema_fast != INVALID_HANDLE) IndicatorRelease(g_handle_ema_fast);
   if(g_handle_ema_slow != INVALID_HANDLE) IndicatorRelease(g_handle_ema_slow);
   if(g_handle_rsi != INVALID_HANDLE) IndicatorRelease(g_handle_rsi);
   if(g_handle_macd != INVALID_HANDLE) IndicatorRelease(g_handle_macd);
   if(g_handle_adx != INVALID_HANDLE) IndicatorRelease(g_handle_adx);
   if(g_handle_mfi != INVALID_HANDLE) IndicatorRelease(g_handle_mfi);
}


void OnTick()
{
   if(!IsNewBar())
      return;

   EvaluateSignalOnClosedBar();
}
