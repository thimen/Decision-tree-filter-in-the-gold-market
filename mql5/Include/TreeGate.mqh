double EvaluateDecisionTreeProbability(
   const double rsi,
   const double adx,
   const double mfi,
   const double ema_gap,
   const double macd_gap,
   const double di_gap,
   const double kst_gap,
   const double ema_vote,
   const double rsi_vote,
   const double macd_vote,
   const double dmi_vote,
   const double kst_vote,
   const double mfi_vote,
   const double buy_vote_count,
   const double sell_vote_count,
   const double signal_direction,
   const double signal_is_buy,
   const double signal_is_sell
)
{
   if (di_gap <= -0.2273920327)
   {
      if (mfi <= 40.2279891968)
      {
         if (mfi <= 37.5724086761)
         {
            return 0.3141025641;
         }
         else
         {
            return 0.5740740741;
         }
      }
      else
      {
         if (ema_gap <= -0.4018857777)
         {
            return 0.3244680851;
         }
         else
         {
            return 0.1757575758;
         }
      }
   }
   else
   {
      if (adx <= 19.3073644638)
      {
         if (rsi <= 56.1263771057)
         {
            return 0.4247311828;
         }
         else
         {
            return 0.2268041237;
         }
      }
      else
      {
         if (macd_gap <= 0.3994308114)
         {
            return 0.5147679325;
         }
         else
         {
            return 0.3538461538;
         }
      }
   }
}
