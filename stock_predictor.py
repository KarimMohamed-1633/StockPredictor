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


#print(company_data)

# ----------------------------------------------- Plotting the stock closing and volume traded ------------------------------------


plt.figure(figsize=(12, 6))


# Plotting the closing price
plt.plot(company_data[stock_symbol].index, company_data[stock_symbol]['Close'], label='Close Price', color='black')
plt.title('Closing Price of ' + stock_symbol)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.ylim(100, 300)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Plotting the volume traded
plt.plot(company_data[stock_symbol].index, company_data[stock_symbol]['Volume'], label='Volume Traded', color='blue')
plt.title('Volume Traded of ' + stock_symbol)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.tight_layout()
plt.show()

#--------------------------------------- Moving Average Predicting Method ---------------------------------------------------------

# Define the list of moving average days
ma_days = [10, 20, 50]

# Iterate over each moving average day
for ma in ma_days:
    # Iterate over each company in company_data
    for company_name, company_df in company_data.items():
        # Calculate moving average for 'Adj Close' column and store it in a new column
        column_name = f'MA - {ma} days'
        company_df[column_name] = company_df['Close'].rolling(window=ma).mean()

# Select the stock data for the specified stock symbol
stock_data = company_data[stock_symbol]

# Plot the 'Adj Close' prices and moving averages for 10, 20, and 50 days
stock_data[['Close', 'MA - 10 days', 'MA - 20 days', 'MA - 50 days']].plot(figsize=(16, 6))

# Set the title of the plot
plt.title(f'{stock_symbol} Stock Analysis')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
