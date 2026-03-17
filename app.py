import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.title("📈 Yahoo Stock Price Forecasting with ARIMA")

# Load Data (relative path)

DATA_FILE = "yahoo_data.xlsx"  
try:
    df = pd.read_excel(DATA_FILE)
except FileNotFoundError:
    st.error(f"File not found: {DATA_FILE}. Upload it to your repo.")
    st.stop()

df.columns = df.columns.str.strip().str.lower()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Detect date and price columns

date_col = next((c for c in df.columns if 'date' in c), None)
if date_col is None:
    st.error("No date column detected.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = next((c for c in df.columns if any(k in c for k in ['close','adj close','price'])), None)
if price_col is None:
    st.error("No price column detected.")
    st.stop()

df.rename(columns={price_col: 'price'}, inplace=True)

# Plot Price

st.subheader("Price Plot")
st.line_chart(df['price'])

# Stationarity Test

def adfuller_test(series):
    return adfuller(series.dropna())[1]

p_original = adfuller_test(df['price'])
st.write(f"ADF Test p-value (Original Series): {p_original:.4f}")

if p_original <= 0.05:
    st.success("Series is stationary")
else:
    st.warning("Series is non-stationary, applying first difference")
    df['price_diff'] = df['price'].diff()
    p_diff = adfuller_test(df['price_diff'])
    st.write(f"ADF Test p-value (Differenced Series): {p_diff:.4f}")
    st.line_chart(df['price_diff'])

# ARIMA Forecast Inputs

st.subheader("ARIMA Forecast Settings")
p = st.number_input("AR term (p)", min_value=0, max_value=5, value=1)
d = st.number_input("Difference term (d)", min_value=0, max_value=2, value=1)
q = st.number_input("MA term (q)", min_value=0, max_value=5, value=1)
steps = st.number_input("Forecast steps", min_value=1, max_value=60, value=10)

# Run Forecast

if st.button("Run Forecast"):
    model = ARIMA(df['price'], order=(p,d,q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    forecast_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq='B'
    )

    forecast_series = pd.Series(forecast.values, index=forecast_index)

    st.subheader("Forecast Plot")
    st.line_chart(pd.concat([df['price'], forecast_series]))

    st.subheader("Forecasted Prices")
    st.dataframe(forecast_series)

    st.write(f"ARIMA({p},{d},{q}) AIC: {model_fit.aic:.2f}")
