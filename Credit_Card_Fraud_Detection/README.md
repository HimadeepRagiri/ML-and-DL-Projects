# Credit Card Fraud Detection

This repository contains code for detecting fraudulent credit card transactions using machine learning, specifically logistic regression.

## Key Files

- `creditcard.csv`: The dataset containing credit card transaction data.
- `Credit_Card_Fraud_Detection.ipynb`: The Python script for data preprocessing, model training, and evaluation.

## Steps to Run

1. Clone this repository.
2. Ensure you have Python installed.
3. Install the required libraries: `pip install pandas numpy scikit-learn`
4. Run the script: `Credit_Card_Fraud_Detection.ipynb`

## Documentation

### Data Loading and Exploration

- Loads the dataset using pandas.
- Displays information about the dataset.
- Analyzes the distribution of legitimate and fraudulent transactions.
- Calculates statistical measures of transaction amounts for both types of transactions.

### Data Preprocessing

- Creates a balanced sample dataset containing both normal and fraudulent transactions.
- Splits the data into features (X) and target (y).
- Splits the data into training and testing sets.

### Model Building and Training

- Creates a logistic regression model.
- Trains the model on the training data.

### Model Evaluation

- Evaluates the model's performance on both training and testing data using accuracy as the metric.

## Contributions

Contributions to this project are welcome. You can suggest improvements, report issues, or create pull requests.
