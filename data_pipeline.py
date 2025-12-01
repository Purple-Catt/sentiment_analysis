import datetime as dt
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd
import yfinance as yf  # yfinance library for Yahoo Finance data [web:47][web:53][web:56][web:59]
from config import TICKERS, START_DATE, END_DATE, NEWS_API_KEY, NEWS_API_ENDPOINT

def download_price_data(tickers=None, start_date=None, end_date=None):
    if tickers is None:
        tickers = TICKERS
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)  # [web:47][web:53][web:56][web:59]
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"].copy()
    df = df.dropna(how="all")
    return df  # index=Date, columns=tickers

def fetch_news_for_ticker(ticker, from_date, to_date, page_size=50):
    """Example using NewsAPI; you must respect their TOS and provide your key."""
    params = {
        "q": ticker,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(NEWS_API_ENDPOINT, params=params)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])
    rows = []
    for a in articles:
        rows.append(
            {
                "ticker": ticker,
                "published_at": a.get("publishedAt"),
                "source": a.get("source", {}).get("name"),
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
            }
        )
    return pd.DataFrame(rows)

def fetch_news_for_universe(tickers=None, months_back=6):
    if tickers is None:
        tickers = TICKERS
    end = dt.date.today()
    start = end - relativedelta(months=months_back)
    all_news = []
    for t in tickers:
        try:
            df_t = fetch_news_for_ticker(
                t, from_date=start.strftime("%Y-%m-%d"), to_date=end.strftime("%Y-%m-%d")
            )
            all_news.append(df_t)
        except Exception as e:
            print(f"News fetch failed for {t}: {e}")
    if not all_news:
        return pd.DataFrame()
    df = pd.concat(all_news, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["date"] = df["published_at"].dt.date
    return df
