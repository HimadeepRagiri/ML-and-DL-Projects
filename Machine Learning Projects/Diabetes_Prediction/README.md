# Diabetes Prediction using Support Vector Machine (SVM)

This repository contains code for predicting diabetes using the Support Vector Machine (SVM) algorithm. The dataset used is `diabetes.csv`, which contains several features related to diabetes diagnosis.

## Key Files

- `diabetes.csv`: Dataset containing diabetes data.
- `Diabetes_Prediction.ipynb`: Python script for data preprocessing, model training, evaluation, and prediction.

## Required Libraries

- pandas
- numpy
- scikit-learn


## Steps to Run

1. Clone this repository: https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
2. Install required libraries:
3. Run the script: `Diabetes_Prediction.ipynb`

## Documentation

### Data Loading and Exploration
- Loads the dataset using pandas.
- Displays summary information and checks for missing values.
- Displays value counts for the target variable (Outcome).

### Data Preprocessing
- Splits data into features (X) and target variable (y).
- Splits data into training and testing sets.
- Performs feature scaling using StandardScaler.

### Model Building and Training
- Creates an SVM classifier with a linear kernel.
- Trains the model on the preprocessed training data.

### Model Evaluation
- Evaluates the model's accuracy on both training and test sets.
- Makes predictions using sample input data.

## Contribution
Contributions to the project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.


