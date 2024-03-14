from statistics import mean
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import sys

# ----------------------------------------------- Extracting Company Data -------------------------------------------------------------

stock_symbol = input('Please Enter Stock Symbol: \n')


#Setting the dates of data 
end = datetime.now()
start = end - timedelta(days=365)

#fetch historic data from apple 
company_data = {}
company_data[stock_symbol] = yf.download(stock_symbol, start, end)

# ----------------------------------------------- Plotting the stock closing and volume traded ------------------------------------

plt.figure(figsize=(12, 6))


# Plotting the closing price
plt.plot(company_data[stock_symbol].index, company_data[stock_symbol]['Adj Close'], label='Close Price', color='black')
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

#--------------------------------------- Simple Moving Average Predicting Method ---------------------------------------------------------

def calculate_sma(company_data, sma_days):
    # Iterate over each moving average day
    for ma in sma_days:
        # Iterate over each company in company_data
        for company_name, company_df in company_data.items():
            # Calculate moving average for 'Adj Close' column and store it in a new column
            column_name = f'MA - {ma} days'
            company_df[column_name] = company_df['Adj Close'].rolling(window=ma).mean()

# Define the list of moving average days
sma_days = [10, 20, 50]

# Call the function to calculate moving averages
calculate_sma(company_data, sma_days)

# Select the stock data for the specified stock symbol
stock_data = company_data[stock_symbol]

# Plot the 'Adj Close' prices and moving averages for 10, 20, and 50 days
stock_data[['Adj Close', 'MA - 10 days', 'MA - 20 days', 'MA - 50 days']].plot(figsize=(16, 6))

# Set the title of the plot
plt.title(f'{stock_symbol} Stock Analysis')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()

# ----------------------------------------- Calculating Daily Returns of stocks --------------------------------------------------------

fig = plt.figure(figsize=(12, 6))


# Calculate daily returns for Apple
company_df = company_data[stock_symbol]
company_df['Adj Close'] = pd.to_numeric(company_df['Adj Close'], errors='coerce')
company_df['Daily Return'] = company_df['Adj Close'].pct_change() * 100
company_df.dropna(subset=['Daily Return'], inplace=True)

#Plot daily return 
stock_data['Daily Return'].plot(legend=True, linestyle='--', marker='o').set_title(f'{stock_symbol} Daily Returns')

fig.tight_layout()
plt.show()


plt.figure(figsize=(10,8))
company_df = company_data[stock_symbol]
plt.hist(company_df['Daily Return'], bins=50)
plt.xlabel('Daily Returns')
plt.ylabel('Counts')
plt.title(f'{stock_symbol}')
plt.tight_layout()
plt.show()


# ----------------------------------------------------- Risk and standard deviation ------------------------------------------------------

mean_of_close = np.mean(company_df['Adj Close'])

# Calculate the standard deviation of the closing prices
std_dev = np.std(company_df['Adj Close'])

# Create a scatter plot of standard deviation against dates
plt.figure(figsize=(12, 6))
plt.scatter(company_df.index, [std_dev] * len(company_df.index), label='Standard Deviation', color='red')
plt.plot(company_df.index, company_df['Adj Close'], label='Close Price', color='black')
plt.title('Standard Deviation of Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------- Training AI Model --------------------------------------------------------------
# print("predicting stock price of apple")
df = yf.download(stock_symbol, start='2012-01-01', end=datetime.now())

data = df.filter(['Close'])
data_set = data.values

#Get number of rows to train model on
training_data_length = int(np.ceil(len(data_set) * 0.98))

# #Scaling the data
# #MinMaxScaler is used to scale the data between 0 and 1. This is used to train many AI models to ensure that all data have the same scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set)

# # Create the training data set 
# # Create the scaled training data set
train_data = scaled_data[0:int(training_data_length), :]
x_train = []
y_train = []

#---------------------------------------------------------- BUIlD TESTING DATA ---------------------------------------------------------------

# # Create the testing data set
# # Create a new array containing scaled values from index 1543 to 2002 
# test_data = scaled_data[training_data_length - 60: , :]
# # Create the data sets x_test and y_test
x_test = []

# #convert to numpy array
x_test = np.array(x_test)

# Adjusting hyperparameters to have more accuracy in predictions
learning_rate = 0.001
epochs = 20
batch_size = 64

# Increase training data length
#training_data_length = int(np.ceil(len(data_set) * 0.98))  # Using 98% of the data for training

test_data = scaled_data[training_data_length - 60:, :]
x_test = []
y_test = data_set[training_data_length:, :]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, :])  # Including all features
    y_train.append(train_data[i, 0])  # Target remains the same (closing price)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Adjusting model architecture
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Adding dropout layer for regularization
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1))

# Compiling the model with the adjusted learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Training the model with more epochs
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

# Getting testing data
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Making predictions with the updated model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculating RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plotting results
train = data[:training_data_length]
valid = data[training_data_length:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

print(valid[['Close', 'Predictions']])
