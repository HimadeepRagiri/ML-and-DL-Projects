# Credit Risk Scoring System

This project involves a credit risk scoring system that utilizes machine learning and deep learning techniques to classify loan statuses as "Fully Paid" or "Charged Off." The dataset undergoes extensive preprocessing, including handling missing values, scaling features, and encoding categorical variables. Finally, a neural network model is trained to predict loan statuses based on input features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results and Evaluation](#results-and-evaluation)
- [Dependencies](#dependencies)
- [key Files](#key-files)
- [Steps To Run](#steps-to-run)

## Project Overview

The dataset contains various financial features such as loan amounts, interest rates, credit scores, and borrower details. After preprocessing, a neural network model is used to classify whether a loan will be fully paid or charged off.

## Dataset Details

The dataset used in this project is a CSV file containing the following columns:

- `loan_amnt`: Loan amount.
- `term`: Duration of the loan (e.g., 36 months, 60 months).
- `int_rate`: Interest rate on the loan.
- `annual_inc`: Borrower's annual income.
- `dti`: Debt-to-income ratio.
- `revol_bal`: Revolving balance on the borrower's credit.
- `loan_status`: Status of the loan (Fully Paid, Charged Off, etc.).

The dataset is preprocessed to remove irrelevant columns, handle missing data, normalize numerical features, and encode categorical variables.

## Data Preprocessing

The preprocessing steps include:

1. **Handling Missing Data**: 
   - Imputation of missing numerical values using the mean.
   - Imputation of missing categorical values using the most frequent value.

2. **Feature Scaling**:
   - Standard scaling for numerical features to normalize them for model training.

3. **Feature Encoding**:
   - Label Encoding for categorical columns with fewer than 10 unique values.
   - One-Hot Encoding for categorical columns with more than 10 unique values.

4. **Splitting the Data**:
   - The data is split into training and testing sets using stratified sampling to maintain the target distribution.

## Model Training

A neural network model is built using the PyTorch library:

- **Architecture**:
    - Input layer with `128` neurons.
    - Hidden layers with appropriate activation functions.
    - Output layer with a single neuron representing binary classification (`0` for Charged Off, `1` for Fully Paid).

- **Hyperparameters**:
    - Input size: The number of features in the dataset.
    - Number of epochs: `15`
    - Batch size: `64`
    - Learning rate: `0.001`

- **Optimizer**: Adam Optimizer is used to minimize the loss function.
- **Loss Function**: Binary Cross-Entropy Loss.

## Results and Evaluation

After training the model, the following metrics are evaluated:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## Dependencies

This project requires the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `tqdm`

Install the required libraries using pip:

## Key Files

- **`Credit_Risk_Scoring_System.ipynb`**: The Jupyter Notebook containing the full code for data preprocessing, model training, and evaluation.
- **`Credit_Risk_Scoring_System.csv`**: The dataset used for training and testing the model.
- **`credit_risk_model.pth`**: The saved model weights after training, which you can use to load the trained model for predictions or further training.

### Dataset Download
The dataset used in this project is available for download here:

[Download Credit_Risk_Scoring_System.csv](https://drive.google.com/file/d/1-5tkL_MyhCIN0itUzoLgfW_J1qeEeIPe/view?usp=sharing)

## Steps to Run
1. Clone this repository.
2. Install the required libraries:
3. Open and run the notebook: `Credit_Risk_Scoring_System.ipynb`.