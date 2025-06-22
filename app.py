import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

st.set_page_config(page_title="Stock LSTM Predictor", layout="wide")


@st.cache_resource
def load_model_and_scaler():
    model = load_model('model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()


def fetch_data(stock_symbol):
    # Pull data till yesterday
    end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download(stock_symbol, start='2012-01-01', end=end_date)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    return df


def plot_chart(df, title):
    ma50 = df['Close'].rolling(window=50).mean()
    ma200 = df['Close'].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Close'], label='Close', color='blue')
    ax.plot(ma50, label='MA50', color='orange')
    ax.plot(ma200, label='MA200', color='green')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_rsi(df):
    rsi = compute_rsi(df['Close'])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rsi, color='purple')
    ax.axhline(70, color='red', linestyle='--')
    ax.axhline(30, color='green', linestyle='--')
    ax.set_title("RSI Indicator")
    st.pyplot(fig)

def predict_next_day(df):
    last_60 = df['Close'].values[-60:].reshape(-1, 1)
    scaled = scaler.transform(last_60)
    input_data = np.reshape(scaled, (1, 60, 1))
    pred = model.predict(input_data)
    return scaler.inverse_transform(pred)[0][0]


st.title(" Stock Price Predictor (LSTM)")

stock = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, NIFTY.BO):", value="AAPL")

if st.button("Predict"):
    df = fetch_data(stock)

    st.subheader(f"{stock} Stock Price Chart")
    plot_chart(df, f"{stock} Close Price with MA50 and MA200")

    st.subheader("RSI Indicator")
    plot_rsi(df)

    st.subheader(" LSTM Prediction")
    next_price = predict_next_day(df)
    st.success(f"Predicted Closing Price for Next Day: **${next_price:.2f}**")
