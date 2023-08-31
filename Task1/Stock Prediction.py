import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices
y_pred = model.predict(X_test)


# Create separate charts

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date
data.sort_values('Date', inplace=True)

# Calculate the number of days since the first date
data['Days'] = (data['Date'] - data['Date'].min()).dt.days



# Create the Open Price vs Close Price chart with dates on the x-axis
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Open Price'], label='Open Price', color='blue')
plt.plot(data['Date'], data['Close Price'], label='Close Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Open Price vs Close Price')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()
plt.tight_layout()
plt.show()


# Actual price vs Predicted price
plt.figure(figsize=(10, 6))
plt.scatter(data.loc[X_test.index, 'Date'], y_test, color='blue', label='Actual Prices')
plt.plot(data.loc[X_test.index, 'Date'], y_pred, color='red', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Market Prediction')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()
plt.tight_layout()
plt.show()



# High Price vs Low Price
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['High Price'], label='High Price', color='green')
plt.plot(data['Date'], data['Low Price'], label='Low Price', color='purple')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('High Price vs Low Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# No. of Shares vs No. of Trades
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['No.of Shares'], label='No. of Shares', color='orange')
plt.plot(data['Date'], data['No. of Trades'], label='No. of Trades', color='brown')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('No. of Shares vs No. of Trades')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Spread High-Low vs Spread Close-Open
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Spread High-Low'], label='Spread High-Low', color='magenta')
plt.plot(data['Date'], data['Spread Close-Open'], label='Spread Close-Open', color='cyan')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.title('Spread High-Low vs Spread Close-Open')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
