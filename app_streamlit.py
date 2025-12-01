import streamlit as st  # Streamlit web app framework [web:48][web:54][web:57][web:60]
import datetime as dt
import pandas as pd

from data_pipeline import download_price_data, fetch_news_for_universe
from sentiment import add_sentiment_to_news
from features import compute_technical_features, aggregate_sentiment_daily, build_ml_dataset
from model import train_xgb_time_series, predict_latest
from trading_logic import generate_equity_signals, map_equity_to_option_idea
from config import TICKERS, MIN_TRAIN_DAYS

st.set_page_config(page_title="Sentiment-Driven Trade Ideas", layout="wide")

st.title("Sentiment & Technicals Trade Idea Generator (v1)")

with st.sidebar:
    st.header("Configuration")
    tickers = st.multiselect("Tickers", TICKERS, default=TICKERS)
    months_back_news = st.slider("Months of news history", 1, 12, 6)
    train_button = st.button("Run full pipeline")

if train_button:
    with st.spinner("Downloading price data..."):
        price_df = download_price_data(tickers=tickers)

    st.write("Price data shape:", price_df.shape)

    with st.spinner("Fetching news & computing sentiment..."):
        news_df = fetch_news_for_universe(tickers=tickers, months_back=months_back_news)
        news_sent_df = add_sentiment_to_news(news_df)

    st.write("News items:", len(news_sent_df))

    with st.spinner("Computing features..."):
        tech_feats = compute_technical_features(price_df)
        sent_daily = aggregate_sentiment_daily(news_sent_df)
        ml_df = build_ml_dataset(price_df, tech_feats, sent_daily)

    st.write("ML dataset rows:", len(ml_df))

    if len(ml_df["date"].unique()) < MIN_TRAIN_DAYS:
        st.error("Not enough history for training; try expanding date range.")
    else:
        with st.spinner("Training XGBoost model..."):
            model, best_acc = train_xgb_time_series(ml_df)
        st.success(f"Model trained. Best cross-validated accuracy: {best_acc:.3f}")

        with st.spinner("Scoring latest data..."):
            latest_preds = predict_latest(ml_df, model)
            signals = generate_equity_signals(latest_preds)
            option_ideas = map_equity_to_option_idea(signals)

        st.subheader("Latest Equity Signals")
        st.dataframe(signals)

        st.subheader("Option Strategy Ideas (Heuristic v1)")
        st.dataframe(option_ideas)

        # Simple visualization for the first ticker
        first_ticker = tickers[0]
        st.subheader(f"Price & Sentiment for {first_ticker}")

        tech_df_t = tech_feats[first_ticker].copy()  # index is already the date
        tech_df_t = tech_df_t.sort_index()

        merged = pd.merge(
            tech_df_t[["close"]],
            ml_df[ml_df["ticker"] == first_ticker][["date", "sent_mean"]]
            .set_index("date"),
            left_index=True,
            right_index=True,
            how="left",
        )

        st.line_chart(merged[["close"]])
        st.line_chart(merged[["sent_mean"]])

else:
    st.info("Configure parameters in the sidebar and click 'Run full pipeline'.")
