# Importing necessary libraries
import yahoo_fin.stock_info as si
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fetch historical data for AMD using yahoo_fin library
# This data includes daily trading metrics such as opening, closing, high, low prices, and volume
amd_stock_data = si.get_data("AMD", start_date="01/01/2023", end_date="11/31/2023")

# Feature Engineering:
# Calculate the daily change in 'adjusted close' prices, which reflects the stock's daily performance
# Convert this change into a binary target variable, where 1 indicates a price increase, and 0 indicates a decrease or no change
amd_stock_data['Price_Diff'] = amd_stock_data['adjclose'].diff()
amd_stock_data['Target'] = (amd_stock_data['Price_Diff'] > 0).astype(int)

# Data Cleaning:
# Remove the first row with NaN values that resulted from the differential operation
amd_stock_data = amd_stock_data.dropna()

# Model Preparation:
# Select intrinsic stock attributes like opening, closing, highs, lows, and volumes as predictors (features)
features = amd_stock_data[['open', 'high', 'low', 'close', 'volume']]
target = amd_stock_data['Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Random Forest Classifier:
# An ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction
# Each tree in the forest is trained on a random subset of the data and features, reducing the risk of overfitting
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation:
# Predict the stock movement on the testing set
predictions = model.predict(X_test)

# Calculate the accuracy of the model on the test data
# Accuracy is the proportion of correctly predicted observations to the total observations
print("Accuracy:", accuracy_score(y_test, predictions))
