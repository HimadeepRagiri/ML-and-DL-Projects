# Big Mart Sales Prediction with XGBRegressor

This project focuses on predicting the sales of items in the Big Mart stores based on various features. The dataset used for this project is from the Big Mart Sales dataset.

## Key Files

- `Big_Mart_Sales.csv`: Dataset containing Big_Mart_Sales data.
- `Big_Mart_Sales_Prediction.ipynb`: Python script for data preprocessing, model training, evaluation, and prediction.

## Overview

The project involves the following steps:

1. **Data Exploration and Cleaning:**
   - Handling missing values by filling them with the mean for 'Item_Weight' and using the mode for 'Outlet_Size'.
   - Exploring and visualizing the distributions of various features.

2. **Data Preprocessing:**
   - Encoding categorical variables using Label Encoding.
   - Splitting the dataset into features (X) and the target variable (y).
   
3. **Model Development:**
   - Using XGBoost regressor to create a predictive model.
   - Splitting the data into training and testing sets.
   - Training the model on the training data and evaluating its performance.

4. **Performance Evaluation:**
   - Calculating the R squared value for both the training and test datasets.

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Steps to Run

1. Clone this repository: https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
2. Install required libraries:
3. Run the script: `Big_Mart_Sales_Prediction.ipynb`

## Contribution
Contributions to the project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.