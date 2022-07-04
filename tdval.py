import yfinance as yf
import numpy as np
#from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from model import *
look_back = 100
GetFacebookInformation = yf.Ticker("BTC-USD")
df = GetFacebookInformation.history(period="6mo")
df = df['Close']
df = df[len(df)-look_back:]
print(df.head())
scaler = MinMaxScaler(feature_range=(-1,1))
df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

td_data = np.expand_dims(df1, axis=0)

td_data = torch.from_numpy(td_data)
print(td_data.shape)
td_val = pred(td_data)
#td_val = scaler.inverse_transform(td_val.numpy())
val = np.array(td_val.detach())
td_val = val.reshape(1,-1)
td_val = scaler.inverse_transform(td_val)
print(td_val)

