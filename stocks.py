import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.title(" Stock Price Prediction App")

# Sidebar
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
future_days = st.sidebar.slider("Days to Predict into the Future", 1, 60, 30)

# Load stock data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found. Please check your inputs.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.line_chart(data['Close'])

    # Preparing data
    df = data[['Close']]
    df['Prediction'] = df['Close'].shift(-future_days)

    X = np.array(df.drop(['Prediction'], axis=1))[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]

    # Model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future
    X_future = df.drop(['Prediction'], axis=1)[-future_days:]
    predictions = model.predict(X_future)

    # Plot
    st.subheader(" Future Price Predictions")
    future_dates = pd.date_range(end_date + timedelta(days=1), periods=future_days)
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    prediction_df.set_index('Date', inplace=True)
    st.line_chart(prediction_df)

    # Display table
    st.write(prediction_df.tail())
