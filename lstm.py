import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="📈 LSTM Stock Price Prediction", layout="wide")
st.title("📈 Apple Stock Price Prediction using LSTM")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.sidebar.button("Train Model"):
    st.subheader(f"Training LSTM Model for {ticker}...")

    # Step 1: Download data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for this ticker and date range.")
    else:
        close_prices = data['Close'].values.reshape(-1, 1)

        # Step 2: Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Step 3: Create dataset
        def create_dataset(dataset, time_step=60):
            x, y = [], []
            for i in range(time_step, len(dataset)):
                x.append(dataset[i-time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(x), np.array(y)

        time_step = 60
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Step 4: Split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Step 5: LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', op
