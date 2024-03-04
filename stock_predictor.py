#Imports 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from pandas_datareader.data import DataReader
#from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
#import tensorflow as tf
import sys


# ----------------------------------------------- Extracting Company Data -------------------------------------------------------------
stock_symbol = "AAPL"

#Setting the dates of data 
end = datetime.now()
start = end - timedelta(days=365)

#fetch historic data from apple 
company_data = {}
company_data[stock_symbol] = yf.download(stock_symbol, start, end)

print(company_data)



# ----------------------------------------------- Plotting the stock closing and volume traded ------------------------------------











#--------------------------------------- Moving Average Predicting Method ---------------------------------------------------------

def calculate_sma(stock_prices, window_size):
    sma_values = []
    for i in range(len(stock_prices) - window_size + 1):
        window_sum = sum(stock_prices[i:i+window_size])
        sma = window_sum / window_size 
        sma_values.append(sma)
    return sma_values


#Extract the closing prices 
#stock_prices = stockData[("Close")]

#Train and test ratios 
#test_ratio = 0.2
#train_ratio = 1 - test_ratio


#divide our historical stock price data into two parts: 
# one part for teaching our prediction model (training), 
# and the other part for testing how good our model is at making predictions (testing).
#train_size = int(train_ratio * len(stock_prices))
#test_size = int(test_ratio * len(stock_prices))


#print(f"train_size: {train_size}")
#print(f"test_size: {test_size}")

#Testing gituhb commit 
#test 2

def print_HeLOO():
    print("Hello")