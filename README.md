# Stock Price Trend Prediction with LSTM

## Project Overview

This project aims to forecast stock closing prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price data and predicts the next day's closing price based on the past 60 days. Technical indicators such as Moving Averages and RSI are also integrated for trend and momentum analysis. A Streamlit dashboard is included to demonstrate predictions and visualize key metrics.

## 🔧 Tools & Technologies Used

- **Python**
- **Jupyter Notebook** — Model training and development
- **VS Code** — Streamlit dashboard
- **Libraries**:
  - `pandas`, `numpy`, `scikit-learn`
  - `keras`, `tensorflow`
  - `yfinance`, `matplotlib`,`plotly`
  - `streamlit`

##  Workflow Summary

### 1. Data Collection
- Pulled historical stock price data using the `yfinance` API (2012 to 2025)
- Focused on the `Close` price column

### 2. Preprocessing
- Applied MinMaxScaler to normalize prices
- Created time-series windows of 60 days to predict the 61st
- Split into training (80%) and test (20%) sets

### 3. Model Building
- Used Sequential Keras model with two LSTM layers and dropout
- Compiled with `mean_squared_error` loss and `adam` optimizer
- Trained for 20 epochs

### 4. Prediction & Evaluation
- Inverse-transformed predictions to get actual price values
- Plotted predicted vs actual closing prices
- Integrated:
  - **50 & 200-day Moving Averages**
  - **RSI (Relative Strength Index)**

### 5. Streamlit Dashboard
- Takes stock symbol as input
- Predicts the next day’s closing price using the most recent 60 days
- Visualizes:
  - Candlestick chart (attempted using Plotly but failed.will do in the future)
  - Moving Averages
  - RSI
  - Predicted next closing price

## Repository Contents

├── stock_modeling.ipynb # Model training and preprocessing
├── model.h5 # Trained LSTM model
├── scaler.pkl # Scaler used during training
├── app.py # Streamlit dashboard
├── assets/ # Screenshots for visual reference
├──project report.pdf # Final project report
└── README.md # This file

## 🔍 Notes

- Yahoo Finance data is returned in USD by default.
- When using `.NS` tickers for Indian stocks (e.g., RELIANCE.NS), values still reflect Yahoo’s currency formatting.
- The model predicts **next day's price**, assuming today’s data is not included in training (for live demo).
- Accuracy is trend-based and not intended for real trading decisions since the the real prices are influenced by more than one factor.

## Example Inputs

- `AAPL`, `TSLA` — US Stocks
- `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS` — Indian Stocks


## 📷 Screenshots

### Actual vs Predicted
![Prediction Chart](<assets/Screenshot 2025-06-23 015005.png>)

### RSI Indicator
![RSI Chart](<assets/Screenshot 2025-06-23 020127.png>)

### Moving Averages
![MA Chart](<assets/Screenshot 2025-06-23 020001.png>)


##  Final Deliverables

- Jupyter Notebook(`stock_modeling.ipynb`)
- Trained model file (`model.h5`)
- Scaler file (`scaler.pkl`)
- Streamlit dashboard(`app.py`)
- 2-page project report (`project report.pdf`)
