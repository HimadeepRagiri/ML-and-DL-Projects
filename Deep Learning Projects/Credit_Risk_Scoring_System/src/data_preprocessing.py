# -*- coding: utf-8 -*-
"""data_preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nDl8Wp3xW1ZY3p6eHLCO1H-LTTnECZwm
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)

def clean_data(dataset):
    # Drop irrelevant columns
    irrelevant_columns = [
        'id', 'member_id', 'url', 'desc', 'zip_code', 'title',
        'emp_title', 'emp_length', 'annual_inc_joint', 'dti_joint',
        'verification_status_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high',
        'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc',
        'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il',
        'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
        'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog',
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
        'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
        'hardship_last_payment_amount', 'disbursement_method', 'debt_settlement_flag',
        'debt_settlement_flag_date', 'settlement_status', 'settlement_date',
        'settlement_amount', 'settlement_percentage', 'settlement_term'
    ]
    dataset = dataset.drop(columns=irrelevant_columns)
    return dataset

def handle_missing_values(dataset):
    # Drop columns with high null values
    threshold = 0.5 * len(dataset)
    high_null_cols = dataset.isnull().sum()[dataset.isnull().sum() > threshold]
    dataset = dataset.drop(columns=high_null_cols.index)

    # Impute numerical and categorical values
    numerical_with_nulls = dataset.select_dtypes(include=['float64', 'int64']).columns
    numerical_with_nulls = numerical_with_nulls[dataset[numerical_with_nulls].isnull().any()]

    imputer_num = SimpleImputer(strategy='mean')
    dataset[numerical_with_nulls] = imputer_num.fit_transform(dataset[numerical_with_nulls])

    categorical_with_nulls = dataset.select_dtypes(include=['object']).columns
    categorical_with_nulls = categorical_with_nulls[dataset[categorical_with_nulls].isnull().any()]
    imputer_cat = SimpleImputer(strategy='most_frequent')
    dataset[categorical_with_nulls] = imputer_cat.fit_transform(dataset[categorical_with_nulls])
    return dataset

def encode_data(dataset):
    dataset = dataset[dataset['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    dataset['loan_status_binary'] = dataset['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

    categorical_columns = dataset.select_dtypes(include=['object']).columns
    one_hot_columns = []
    for col in categorical_columns:
        unique_values = dataset[col].nunique()
        if unique_values <= 10:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
        else:
            one_hot_columns.append(col)
    if one_hot_columns:
        dataset = pd.get_dummies(dataset, columns=one_hot_columns, drop_first=True)
    return dataset

def split_and_scale_data(dataset):
    X = dataset.drop(columns=['loan_status'])
    y = dataset['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def optimize_dtype(df):
    if isinstance(df, pd.Series):  # Check if it's a Series
        col_type = df.dtype
        if col_type == 'float64':
            df = df.astype('float32')
        elif col_type == 'int64':
            df = df.astype('int32')
        elif col_type == 'bool':
            df = df.astype('int8')
    elif isinstance(df, pd.DataFrame):  # Check if it's a DataFrame
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type == 'float64':
                df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                df[col] = df[col].astype('int32')
            elif col_type == 'bool':
                df[col] = df[col].astype('int8')
    return df