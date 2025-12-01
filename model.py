import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

FEATURE_COLS = [
    "ret_1d",
    "ma_5",
    "ma_20",
    "ma_ratio_5_20",
    "vol_10",
    "rsi_14",
    "sent_mean",
    "sent_count",
    "sent_pos_frac",
    "sent_neg_frac",
]

def train_xgb_time_series(df_ml):
    df_ml = df_ml.sort_values(["date", "ticker"])
    X = df_ml[FEATURE_COLS].values
    y = df_ml["label_up"].values

    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = -1.0

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        preds = (model.predict_proba(X_val)[:, 1] > 0.5).astype(int)
        acc = accuracy_score(y_val, preds)
        if acc > best_score:
            best_score = acc
            best_model = model

    return best_model, best_score

def predict_latest(df_ml, model):
    latest_dates = df_ml.groupby("ticker")["date"].max()
    rows = []
    for ticker, d in latest_dates.items():
        row = df_ml[(df_ml["ticker"] == ticker) & (df_ml["date"] == d)].iloc[-1]
        X = row[FEATURE_COLS].values.reshape(1, -1)
        prob_up = float(model.predict_proba(X)[0, 1])
        rows.append(
            {
                "ticker": ticker,
                "date": d,
                "prob_up": prob_up,
            }
        )
    return pd.DataFrame(rows)
