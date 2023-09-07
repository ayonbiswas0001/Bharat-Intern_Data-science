import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV file
data = pd.read_csv('Tata-steel.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date
data.sort_values('Date', inplace=True)

# Calculate the number of days since the first date
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Prepare data for training
X = data[['Days']]
y = data['Close Price']  # Use 'Close Price' for prediction

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data to be suitable for LSTM
X_train_reshaped = X_train_scaled.reshape(-1, 1, 1)
X_test_reshaped = X_test_scaled.reshape(-1, 1, 1)

# Build an LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32)

# Predict stock prices using the trained LSTM model
y_pred_lstm = model.predict(X_test_reshaped)

# Create separate charts

# Open Price vs Close Price with LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Open Price'], label='Open Price', color='blue')
plt.plot(data['Date'], data['Close Price'], label='Close Price', color='red')
plt.plot(data.loc[X_test.index, 'Date'], y_pred_lstm, color='green', label='LSTM Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Open Price vs Close Price with LSTM Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# High Price vs Low Price with LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['High Price'], label='High Price', color='green')
plt.plot(data['Date'], data['Low Price'], label='Low Price', color='purple')
plt.plot(data.loc[X_test.index, 'Date'], y_pred_lstm, color='red', label='LSTM Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('High Price vs Low Price with LSTM Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# No. of Shares vs No. of Trades with LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['No.of Shares'], label='No. of Shares', color='orange')
plt.plot(data['Date'], data['No. of Trades'], label='No. of Trades', color='brown')
plt.plot(data.loc[X_test.index, 'Date'], y_pred_lstm, color='red', label='LSTM Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('No. of Shares vs No. of Trades with LSTM Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Spread High-Low vs Spread Close-Open with LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Spread High-Low'], label='Spread High-Low', color='magenta')
plt.plot(data['Date'], data['Spread Close-Open'], label='Spread Close-Open', color='cyan')
plt.plot(data.loc[X_test.index, 'Date'], y_pred_lstm, color='red', label='LSTM Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.title('Spread High-Low vs Spread Close-Open with LSTM Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Deliverable Quantity vs % Deli. Qty to Traded Qty with LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Deliverable Quantity'], label='Deliverable Quantity', color='blue')
plt.plot(data['Date'], data['% Deli. Qty to Traded Qty'], label='% Deli. Qty to Traded Qty', color='green')
plt.plot(data.loc[X_test.index, 'Date'], y_pred_lstm, color='red', label='LSTM Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Deliverable Quantity vs % Deli. Qty to Traded Qty with LSTM Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# Actual price vs Predicted price using LSTM
plt.figure(figsize=(10, 6))
plt.scatter(data.loc[X_test.index, 'Date'], y_test, color='blue', label='Actual Prices')
plt.plot(data.loc[X_test.index, 'Date'], y_pred_lstm, color='green', label='LSTM Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Market Prediction (LSTM)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
