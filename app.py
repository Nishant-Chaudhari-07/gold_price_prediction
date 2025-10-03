import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title='Gold Price Prediction', layout='centered')

st.title('📈 Gold Price Prediction')
st.write('Enter engineered features (lags, rolling stats, returns) and get a predicted price. These values are usually computed from recent price history.')

MODEL_PATH = 'gold_model.pkl'
META_PATH = 'model_meta.json'

model = joblib.load(MODEL_PATH)
meta = json.load(open(META_PATH))
FEATURES = meta['features_in_order']

st.caption('Model: ' + meta['best_model_name'])

with st.form('pred_form'):
    vals = []
    for feat in FEATURES:
        if feat.startswith('lag_'):
            v = st.number_input(f"{feat} (prev-day price features)", value=1800.0, step=1.0, format='%.4f')
        elif feat.startswith('rollmean_') or feat.startswith('rollstd_'):
            v = st.number_input(f"{feat} (rolling stats)", value=1800.0, step=1.0, format='%.4f')
        elif feat.startswith('ret_'):
            v = st.number_input(f"{feat} (returns)", value=0.0, step=0.0001, format='%.6f')
        elif feat in ['rsi_14', 'atr_14']:
            v = st.number_input(f"{feat}", value=50.0, step=0.1, format='%.4f')
        elif feat in ['year', 'month', 'dayofweek']:
            default = 2024 if feat == 'year' else (1 if feat == 'month' else 0)
            v = st.number_input(f"{feat}", value=float(default), step=1.0, format='%.0f')
        else:
            v = st.number_input(feat, value=0.0, step=0.1, format='%.4f')
        vals.append(v)

    submitted = st.form_submit_button('Predict')

if submitted:
    X = pd.DataFrame([vals], columns=FEATURES)
    pred = float(model.predict(X)[0])
    st.success(f'Predicted Gold Price: ${pred:,.2f}')
