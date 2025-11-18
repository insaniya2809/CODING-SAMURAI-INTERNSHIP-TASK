# Project 5 - Time Series Forecasting for Stock Prices

import os
import warnings
warnings.filterwarnings("ignore")


os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)


import matplotlib
matplotlib.use('TkAgg')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


ticker = "AAPL"
df = yf.download(ticker, period="5y", interval="1d")
df = df[['Close']].rename(columns={'Close': 'close'})
df.index = pd.to_datetime(df.index)
df.to_csv(f"data/{ticker}_data.csv")

print("✅ Data downloaded. Shape:", df.shape)
print(df.head())


plt.figure(figsize=(10,4))
plt.plot(df['close'])
plt.title(f"{ticker} Closing Price (5 Years)")
plt.grid(True)
plt.show(block=True)


df_reset = df.reset_index(drop=True)


train_size = int(len(df_reset) * 0.8)
train = df_reset['close'][:train_size]
test = df_reset['close'][train_size:]

print("\nTrain size:", len(train))
print("Test size:", len(test))


adf = adfuller(train.dropna())
print("\n✅ ADF Statistic:", adf[0])
print("✅ p-value:", adf[1])


model = SARIMAX(train, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

print("\n✅ Model trained successfully")


start_idx = train_size
end_idx = train_size + len(test) - 1

pred = model_fit.predict(start=start_idx, end=end_idx)


rmse = np.sqrt(mean_squared_error(test, pred))
mae = mean_absolute_error(test, pred)

print("\n📊 Evaluation Metrics")
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))


plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual Test")
plt.plot(pred.index, pred, label="Predicted")
plt.title(f"{ticker} Price Forecast (ARIMA)")
plt.legend()
plt.grid(True)
plt.show(block=True)


joblib.dump(model_fit, f"model/{ticker}_arima_model.joblib")
print("\n✅ Model saved successfully")

input("\nPress ENTER to exit…")
