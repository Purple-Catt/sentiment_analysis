import pandas as pd

def generate_equity_signals(pred_df, threshold_buy=0.55, threshold_sell=0.45):
    signals = []
    for _, row in pred_df.iterrows():
        if row["prob_up"] >= threshold_buy:
            signal = "BUY"
        elif row["prob_up"] <= threshold_sell:
            signal = "SELL"
        else:
            signal = "HOLD"
        signals.append(
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "prob_up": row["prob_up"],
                "signal": signal,
            }
        )
    return pd.DataFrame(signals)

def map_equity_to_option_idea(signals_df):
    ideas = []
    for _, r in signals_df.iterrows():
        if r["signal"] == "BUY":
            idea = "Bullish: consider buying a 1-2 month ATM call or bull call spread."
        elif r["signal"] == "SELL":
            idea = "Bearish: consider buying a 1-2 month ATM put or bear put spread."
        else:
            idea = "Neutral: consider no trade or income strategies (e.g., covered calls) if long."
        ideas.append(
            {
                "ticker": r["ticker"],
                "date": r["date"],
                "prob_up": r["prob_up"],
                "signal": r["signal"],
                "option_idea": idea,
            }
        )
    return pd.DataFrame(ideas)
