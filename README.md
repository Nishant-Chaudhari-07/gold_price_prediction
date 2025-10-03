# 📈 Gold Price Prediction using Machine Learning

This project builds a **machine learning–driven forecasting system** for daily gold prices using historical data (2013–2023).  
It demonstrates the full lifecycle of a predictive analytics solution — from **data cleaning and feature engineering** to **model training, evaluation, and deployment** via a Streamlit app.


## 🚀 Project Overview
- **Objective**: Predict next-day gold prices to support financial analysis, risk management, and trading decisions.  
- **Data**: 10 years of daily gold price history (2013–2023).  
- **Tech stack**: Python, Scikit-Learn, Pandas, NumPy, Altair, Streamlit.  
- **Deployment**: Interactive web app deployed on Streamlit Cloud.

The app provides:  
- 📊 Daily next-day gold price forecasts  
- 🔄 Backtesting with walk-forward validation  
- 📉 Model performance metrics (RMSE, MAE, R²)  
- 📋 Residuals analysis with plain-English explanations  


## ✨ Key Features
- **End-to-End ML Pipeline**
  - Data preprocessing & cleaning
  - Feature engineering: lag features, rolling means/volatilities, returns, RSI, ATR
  - Hyperparameter tuning & cross-validation
  - Regression & ensemble models
- **Interactive Dashboard**
  - Fixed dataset (no manual upload required)
  - Dynamic metrics that update with the backtest window
  - Clear, plain-English summaries for non-technical users
  - Charts: gold price trends, model predictions, residuals
- **Reproducible Deployment**
  - Model serialized in `.pkl` file
  - Metadata stored in `.json` for feature alignment and model info
  - Streamlit app for instant predictions


## 📂 File Structure
```bash
Gold-Price-Prediction/
│
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── gold_model.pkl              # Trained ML model (serialized)
├── model_meta.json             # Metadata: features, model name, etc.
├── Gold Price (2013-2023).csv  # Historical dataset
│
├── notebooks/
│   └── training.ipynb          # Jupyter/Colab notebook: data prep & model training
│
├── README.md                   # Project documentation (this file)

```

## 📊 Model Performance

R² ≈ 0.92–0.99 depending on the backtest window.

RMSE (Root Mean Squared Error): ~$3–$5 typical error.

MAE (Mean Absolute Error): ~$2–$4 average daily miss.

Residual plots show errors are mostly random and centered near 0, meaning the model generalizes well without systematic bias.

## 💡 Use Cases

Finance & Trading: Support trading strategies with predictive insights.

Risk Management: Anticipate volatility for hedging.

Academic & Research: Demonstration of applied machine learning in time-series forecasting.

Portfolio Project: Showcases end-to-end data analytics, model deployment, and storytelling.
