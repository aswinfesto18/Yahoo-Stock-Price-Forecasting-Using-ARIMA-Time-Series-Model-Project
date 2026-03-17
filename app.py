import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.title("📈 Yahoo Stock Price Forecasting with ARIMA")

# Load Excel

FILE_PATH = r"D:\Resume Project\Projects\Yahoo Stock Price Forecasting Using ARIMA Time Series Model Project\yahoo_data.xlsx"

df = pd.read_excel(FILE_PATH)

df.columns = df.columns.str.strip().str.lower()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Detect Date Column

date_col = None
for col in df.columns:
    if 'date' in col:
        date_col = col
        break

if not date_col:
    st.error("No date column detected. Make sure your file has a date column.")
else:
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # Detect Price Column
    price_keywords = ['close', 'adj close', 'price']
    price_col = None
    for col in df.columns:
        for key in price_keywords:
            if key in col:
                price_col = col
                break
        if price_col:
            break

    if not price_col:
        st.error("No price column detected. Make sure your file has a price column like 'Close'.")
    else:
        df.rename(columns={price_col: 'price'}, inplace=True)

        st.subheader("Price Plot")
        st.line_chart(df['price'])

        # Stationarity Test
       
        def adfuller_test(series):
            result = adfuller(series.dropna())
            return result[1]

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

        # ARIMA Forecast
       
        st.subheader("ARIMA Forecast")
        p = st.number_input("AR term (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Difference term (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA term (q)", min_value=0, max_value=5, value=1)
        steps = st.number_input("Forecast steps", min_value=1, max_value=60, value=10)

        if st.button("Run Forecast"):
            model = ARIMA(df['price'], order=(p,d,q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)

            forecast_index = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='B'  # business days
            )

            forecast_series = pd.Series(forecast.values, index=forecast_index)

            st.line_chart(pd.concat([df['price'], forecast_series]))
            st.write("Forecasted Prices:")
            st.dataframe(forecast_series)

            st.write(f"ARIMA({p},{d},{q}) AIC: {model_fit.aic:.2f}")