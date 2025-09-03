# Stock Market Analysis and Forecasting using LSTM (PyTorch)

## Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network model to analyze and forecast stock market trends based on historical stock data. The model predicts the next day’s closing price using past 60 days of stock features (Open, High, Low, Close, Volume). The implementation uses PyTorch for building and training the LSTM model and `yfinance` to download historical stock data.

---

## Features

- Download and preprocess 5+ years of historical stock data (OHLCV format).
- Scale data using Min-Max normalization.
- Prepare sequential data suitable for LSTM input.
- Build a multi-layer LSTM model in PyTorch.
- Train the model with MSE loss and Adam optimizer.
- Evaluate model performance using Root Mean Squared Error (RMSE).
- Visualize actual vs predicted stock prices.

---

## Requirements

- Python 3.7+
- PyTorch
- yfinance
- scikit-learn
- numpy
- pandas
- matplotlib

You can install the required packages using:

```bash
pip install torch yfinance scikit-learn numpy pandas matplotlib
