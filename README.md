# Stock Price Predictor Using Machine Learning

## Project Overview
This project aims to predict the future movement of stock prices using historical data. The focus is on AMD (Advanced Micro Devices) stock, utilizing a machine learning model to forecast whether the stock price will rise or fall on the following day. The project applies various data processing techniques, feature engineering, and a machine learning algorithm for predictive analysis.

## Technologies Used
- **Python 3**
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `yahoo_fin`, `ta` (technical analysis)
- **Machine Learning Algorithm**: `HistGradientBoostingClassifier`

## Features
- Historical stock data retrieval from Yahoo Finance (`yahoo_fin` library)
- Feature engineering with technical indicators using the `ta` library
- Data preprocessing and imputation of missing values
- Training a `HistGradientBoostingClassifier` model to predict stock price movements
- Accuracy evaluation of the predictive model

## Data
The data consists of historical stock prices for AMD, including the following attributes for each trading day:
- `Open`
- `High`
- `Low`
- `Close`
- `Adjusted Close`
- `Volume`

Additional technical indicators were generated as features for the model, including moving averages, RSI, MACD, and others.

## Model
The project uses the `HistGradientBoostingClassifier` from scikit-learn, an effective machine learning algorithm for classification tasks. The model was trained on a subset of the historical data and evaluated on a separate testing set to assess its predictive accuracy.

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
