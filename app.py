import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="Gold Price Prediction", layout="wide")

# =============================
# Config — FIXED DATA SOURCE
# =============================
# Put your CSV in the repo root (or a /data folder) and set the path here.
# Example options:
# DATA_PATH = "Gold Price (2013-2023).csv"
# DATA_PATH = "data/gold_prices.csv"
DATA_PATH = "Gold Price (2013-2023).csv"

# =============================
# Load model + metadata
# =============================
@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load("gold_model.pkl")
    meta = json.load(open("model_meta.json"))
    return model, meta

model, meta = load_assets()
FEATURES = meta.get("features_in_order", [])
BEST_MODEL_NAME = meta.get("best_model_name", "(unknown)")

# =============================
# Helpers: parsing & features
# =============================
NUMERIC_CANDIDATES = [
    "open", "high", "low", "close", "adj_close", "price",
    "close_price", "closing_price", "gold_price", "gold_price_usd",
    "volume"
]

def coerce_numeric_col(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = cleaned.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # coerce likely numeric cols
    for c in [c for c in NUMERIC_CANDIDATES if c in df.columns]:
        df[c] = coerce_numeric_col(df[c])
    # parse date column
    date_cols = [c for c in df.columns if "date" in c]
    if not date_cols:
        raise ValueError("No date-like column found. Column name should contain 'date'.")
    dcol = date_cols[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).drop_duplicates(subset=[dcol]).reset_index(drop=True)
    # rename target close column to 'close'
    price_cols = [c for c in [
        "close", "adj_close", "price", "close_price", "closing_price", "gold_price", "gold_price_usd"
    ] if c in df.columns]
    if not price_cols:
        raise ValueError("No close/price column found.")
    if price_cols[0] != "close":
        df = df.rename(columns={price_cols[0]: "close"})
    return df.rename(columns={dcol: "date"})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = df[["date", "close"]].copy()
    fe["year"] = fe["date"].dt.year
    fe["month"] = fe["date"].dt.month
    fe["dayofweek"] = fe["date"].dt.dayofweek

    for L in [1, 3, 5, 7, 14, 21, 30]:
        fe[f"lag_{L}"] = fe["close"].shift(L)

    for W in [3, 5, 7, 14, 21, 30]:
        fe[f"rollmean_{W}"] = fe["close"].rolling(W).mean()
        fe[f"rollstd_{W}"] = fe["close"].rolling(W).std()

    fe["ret_1"] = fe["close"].pct_change(1)
    fe["ret_7"] = fe["close"].pct_change(7)
    fe["ret_14"] = fe["close"].pct_change(14)

    # RSI 14
    delta = fe["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    fe["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR 14 (requires OHLC). If original df has it, compute.
    if set(["high", "low", "close"]).issubset(df.columns):
        tmp = df[["date", "high", "low", "close"]].copy().sort_values("date")
        prev_close = tmp["close"].shift(1)
        tr = pd.concat([
            (tmp["high"] - tmp["low"]),
            (tmp["high"] - prev_close).abs(),
            (tmp["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        fe["atr_14"] = tr.rolling(14).mean()

    fe = fe.dropna().reset_index(drop=True)
    return fe

# =============================
# Load fixed CSV and prepare features
# =============================
@st.cache_data(show_spinner=False)
def load_data(path: str):
    base = pd.read_csv(path)
    base = standardize_columns(base)
    fe = add_features(base)
    return base, fe

try:
    base, fe = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load fixed dataset at '{DATA_PATH}': {e}")
    st.stop()

# =============================
# UI Header
# =============================
st.title("Gold Price Prediction – Fixed Dataset")
st.caption(f"Model: {BEST_MODEL_NAME} · Data: {DATA_PATH}")

# =============================
# Data Preview & Chart
# =============================
st.subheader("Data Preview")
st.dataframe(base.tail(10), use_container_width=True)

st.subheader("Price Chart")
st.line_chart(base.set_index("date")["close"])  # Streamlit built-in line chart

# =============================
# Predict next day after the last date in the file
# =============================
st.subheader("Predict Next-Day Price")
last_date = base["date"].max().date()
pred_date = last_date + timedelta(days=1)

latest_row = fe.iloc[[-1]].copy()
X_latest = latest_row[[c for c in FEATURES if c in latest_row.columns]].copy()

if set(FEATURES).issubset(X_latest.columns):
    pred_val = float(model.predict(X_latest)[0])
    st.success(f"Predicted Close for {pred_date.isoformat()}: ${pred_val:,.2f}")
else:
    missing_feats = [c for c in FEATURES if c not in X_latest.columns]
    st.error(f"Model expects features not present in engineered data: {missing_feats}")

# =============================
# Backtest on recent window
# =============================
st.subheader("Model Backtest (walk-forward on recent data)")
window_days = st.slider("Backtest window (days)", min_value=60, max_value=730, value=365, step=30)

fe_slice = fe.iloc[-window_days:].copy()
X_cols = [c for c in FEATURES if c in fe_slice.columns]

if not X_cols:
    st.warning("No overlapping features between data and model. Cannot backtest.")
else:
    X_bt = fe_slice[X_cols]
    y_bt = fe_slice["close"].values
    preds_bt = model.predict(X_bt)

    bt_df = pd.DataFrame({
        "date": fe_slice["date"].values,
        "actual": y_bt,
        "predicted": preds_bt
    }).set_index("date")

    # Metrics
    rmse = float(np.sqrt(np.mean((bt_df["actual"] - bt_df["predicted"])**2)))
    mae = float(np.mean(np.abs(bt_df["actual"] - bt_df["predicted"])))
    r2 = float(1 - ((bt_df["actual"] - bt_df["predicted"])**2).sum() / ((bt_df["actual"] - bt_df["actual"].mean())**2).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("R²", f"{r2:,.3f}")

    st.line_chart(bt_df[["actual", "predicted"]])

    st.markdown("**Residuals (Actual - Predicted)**")
    res_df = pd.DataFrame({
        "date": bt_df.index,
        "residual": (bt_df["actual"] - bt_df["predicted"]).values
    }).set_index("date")
    st.line_chart(res_df)

# =============================
# Sidebar: About & Inputs Explained
# =============================
with st.sidebar:
    st.header("About this app")
    st.markdown(
        """
        **Fixed data source**: The app always uses a bundled CSV (see the file path shown on top).

        **What the model uses (auto-computed):**
        - **Lag_N**: previous N-day prices to capture recent levels
        - **Rolling Mean/Std**: trend & volatility over past N days
        - **Returns**: % change over 1/7/14 days
        - **RSI(14)**: momentum (0–100), ~50 neutral
        - **ATR(14)**: volatility from True Range (needs High/Low/Close)

        **How to update data**: replace the CSV file in the repo with newer history,
        keeping the same columns. The app will recompute features automatically.
        """
    )
    st.divider()
    st.caption("Built for portfolio/demo purposes. Not financial advice.")
