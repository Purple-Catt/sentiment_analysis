import numpy as np
import pandas as pd
from config import LABEL_HORIZON_DAYS

def compute_technical_features(price_df):
    feats = {}
    for col in price_df.columns:
        s = price_df[col].dropna()
        df = pd.DataFrame(index=s.index)
        df["close"] = s
        df["ret_1d"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_ratio_5_20"] = df["ma_5"] / df["ma_20"]
        df["vol_10"] = df["ret_1d"].rolling(10).std()
        df["rsi_14"] = compute_rsi(df["close"], window=14)
        feats[col] = df
    return feats  # dict[ticker] -> DataFrame

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=window).mean()
    roll_down = down.ewm(span=window).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def aggregate_sentiment_daily(df_news_sent):
    if df_news_sent.empty:
        return {}
    grouped = df_news_sent.groupby(["ticker", "date"])
    out = {}
    for (ticker, date), g in grouped:
        key = (ticker, pd.to_datetime(date))
        out[key] = {
            "sent_mean": g["sentiment_score"].mean(),
            "sent_median": g["sentiment_score"].median(),
            "sent_count": len(g),
            "sent_pos_frac": (g["sentiment_score"] > 0).mean(),
            "sent_neg_frac": (g["sentiment_score"] < 0).mean(),
        }
    return out

def build_ml_dataset(price_df, tech_feats_dict, daily_sent_dict):
    rows = []
    for ticker, tech_df in tech_feats_dict.items():
        for date, row in tech_df.iterrows():
            key = (ticker, date.normalize())
            sent = daily_sent_dict.get(key, None)
            sent_mean = sent["sent_mean"] if sent else 0.0
            sent_count = sent["sent_count"] if sent else 0
            sent_pos_frac = sent["sent_pos_frac"] if sent else 0.0
            sent_neg_frac = sent["sent_neg_frac"] if sent else 0.0

            try:
                future_idx = tech_df.index.get_loc(date) + LABEL_HORIZON_DAYS
                if future_idx < len(tech_df.index):
                    future_price = tech_df.iloc[future_idx]["close"]
                    today_price = row["close"]
                    future_ret = (future_price - today_price) / today_price
                    label_up = 1 if future_ret > 0 else 0
                else:
                    continue
            except Exception:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": row["close"],
                    "ret_1d": row["ret_1d"],
                    "ma_5": row["ma_5"],
                    "ma_20": row["ma_20"],
                    "ma_ratio_5_20": row["ma_ratio_5_20"],
                    "vol_10": row["vol_10"],
                    "rsi_14": row["rsi_14"],
                    "sent_mean": sent_mean,
                    "sent_count": sent_count,
                    "sent_pos_frac": sent_pos_frac,
                    "sent_neg_frac": sent_neg_frac,
                    "label_up": label_up,
                }
            )
    df = pd.DataFrame(rows)
    df = df.dropna()
    return df
