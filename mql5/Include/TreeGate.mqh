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
      if (rsi <= 44.1779346466)
      {
         if (mfi <= 45.7550163269)
         {
            return 0.4347826087;
         }
         else
         {
            return 0.2631578947;
         }
      }
      else
      {
         if (kst_gap <= 0.6974101067)
         {
            return 0.1913043478;
         }
         else
         {
            return 0.3620689655;
         }
      }
   }
   else
   {
      if (adx <= 18.4278078079)
      {
         if (adx <= 16.8642158508)
         {
            return 0.3919597990;
         }
         else
         {
            return 0.1800000000;
         }
      }
      else
      {
         if (macd_gap <= 0.3443053365)
         {
            return 0.5161290323;
         }
         else
         {
            return 0.3863636364;
         }
      }
   }
}
