# Gold Price Prediction using Random Forest Regressor

This repository contains code for predicting the price of gold using the Random Forest Regressor algorithm. The dataset used is `gold_price_data.csv`, which contains historical data related to gold prices and other factors influencing gold prices.

## Key Files

- `gold_price_data.csv`: Dataset containing gold price data.
- `Gold_Price_Prediction.ipynb`: Python script for data preprocessing, model training, evaluation, and prediction.

## Required Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Steps to Run

1. **Clone this repository:**
2. **Install required libraries:**
3. **Run the script:**

## Documentation

### Data Collection and Preprocessing
- Loads the dataset using pandas.
- Displays summary information and checks for missing values.
- Checks the number of missing values.
- Displays statistical measures of the data.
- Finds the correlation between features and the target variable (GLD).

### Data Visualization
- Visualizes the correlation using a heatmap.
- Plots the distribution of the GLD Price.

### Data Splitting
- Splits the data into features (X) and the target variable (y).
- Splits the data into training and testing sets.

### Feature Scaling
- Performs feature scaling using StandardScaler.

### Model Building and Training
- Creates a RandomForestRegressor model.
- Trains the model on the preprocessed training data.

### Model Evaluation
- Predicts the gold prices on the test data.
- Calculates the R-squared error.
- Compares the actual values and predicted values in a plot.

## Contribution
Contributions to the project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.