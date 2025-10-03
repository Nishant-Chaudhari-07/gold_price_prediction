# Gold Price Prediction (2013–2023)

**Goal:** Predict daily gold closing price using lag, rolling, returns, RSI and (if available) ATR features.

## Data
Use your CSV (e.g., Gold Price (2013-2023).csv) with at least **date** and **close/price** columns. Optional: **open/high/low/volume**.

## Methods
- Time-aware split (80/20)
- Models: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- TimeSeriesSplit CV (you can reduce folds in Colab for speed)
- Metrics: RMSE, MAE, R²
- Baseline: Naïve (price_t ≈ price_(t-1))

## Results
See model_meta.json for metrics and the chosen best model.

## Run Streamlit App
    pip install -r requirements.txt
    streamlit run app.py

## Files
- gold_model.pkl — trained model
- model_meta.json — metadata (features, metrics)
- app.py — Streamlit UI
- requirements.txt — Python deps
