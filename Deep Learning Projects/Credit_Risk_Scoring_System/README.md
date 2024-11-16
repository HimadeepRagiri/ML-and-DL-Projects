# Credit Risk Scoring System

This project involves a credit risk scoring system that utilizes machine learning and deep learning techniques to classify loan statuses as "Fully Paid" or "Charged Off." The dataset undergoes extensive preprocessing, including handling missing values, scaling features, and encoding categorical variables. Finally, a neural network model is trained to predict loan statuses based on input features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results and Evaluation](#results-and-evaluation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
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
- `scikit-learn`
- `torch`
- `tqdm`

Install the required libraries using `pip install -r requirements.txt`

## Project Structure

- `data/`: This folder contains a **`dataset_info.md`** file with instructions on how to download the dataset from Kaggle. The raw data files are not included in the repository due to size constraints.
- `notebooks/`: Contains Jupyter Notebooks with exploratory data analysis (EDA), model training, and evaluation.
- `src/`: Contains Python scripts for data preprocessing, model building, and utility functions.
- `main.py`: Main script to execute the model pipeline.
- `model.pth`: Saved model checkpoint (trained model) for inference or further training.

## Steps to Run

Follow these steps to run the project on your local machine:

1. **Clone the Repository**:
   - Clone the project repository to your local machine:
     ```bash
     git clone <repository_url>
     ```

2. **Install Dependencies**:
   - Install the required libraries by running the following command:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset**:
   - Ensure that you have the dataset (CSV file) in the `data/` directory. If not, download the dataset from the source (if available) and place it in the `data/` folder.

4. **Run the Jupyter Notebooks** (Optional):
   - If you prefer to explore the data and model in a Jupyter Notebook, navigate to the `notebooks/` directory and open the notebook files (`EDA`, `Model_Training`, etc.) in Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - The notebooks include:
     - **Exploratory Data Analysis (EDA)**: For visualizing the dataset, understanding distributions, and finding patterns.
     - **Model Training**: For training the neural network and evaluating its performance.

5. **Running the Model Training Using Python Files**:

   - The project consists of different Python files that handle different stages of the pipeline: preprocessing, model building, training, and evaluation. Here's how to run everything by using them:
   
     1. **Data Preprocessing**:
        - The preprocessing steps are defined in the `src/data_preprocessing.py` file. You can import and call the functions from this file in `main.py` to preprocess the dataset.
        - It handles:
          - Missing data imputation.
          - Feature scaling (standardization).
          - Encoding categorical features (label and one-hot encoding).
          - Splitting data into training and testing sets.

     2. **Utility Functions**:
        - The `src/utils.py` file contains utility functions such as the `load_checkpoint()` function for loading saved models and checkpoints. It also contains other helper functions for logging and saving models.

     3. **Model Building**:
        - The model architecture is defined in the `src/model.py` file, where a neural network is created using the PyTorch library.
        - It defines the architecture (input layer, hidden layers, output layer) and the forward pass method.
        - You can modify this file to experiment with different network architectures if needed.
      
     4. **Training and Evaluation**:
        - The main model training logic is handled in `main.py`. This script:
          - Loads and preprocesses the dataset.
          - Initializes the model, optimizer, and loss function.
          - Trains the model by calling the functions from `src/model.py` and `src/data_preprocessing.py`.
          - Evaluates the model's performance using various metrics like accuracy, precision, recall, and F1 score.
          - Saves the trained model checkpoint.

     5. **Loading a Saved Model**:
        - If you want to load a previously trained model and make predictions, you can use the `load_checkpoint()` function from `src/utils.py` in `main.py`. This function will load the model's state, optimizer state, and training epoch.
        - Example:
          ```python
          checkpoint_path = 'Deep Learning Projects/Credit_Risk_Scoring_System/model.pth'
          model, optimizer, scheduler, epoch, loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
          ```

6. **Run the `main.py` Script**:
   - To train the neural network model, run the `main.py` script:
     ```bash
     python main.py
     ```
   - The script will:
     - Preprocess the data using the functions defined in `src/data_preprocessing.py`.
     - Train the neural network model using the architecture defined in `src/model.py`.
     - Evaluate the trained model and print the results (accuracy, precision, recall, etc.).
     - Save the model checkpoint.

7. **Model Inference**:
   - Once the model is trained and the checkpoint is saved, you can use the trained model for inference (predicting loan statuses on new data).
   - To load the model and use it for inference, you can use the `load_checkpoint()` function in your inference code.
   
8. **Evaluate the Model**:
   - After training, the evaluation metrics (accuracy, precision, recall, F1 score, etc.) will be printed to the console. The confusion matrix and other visualizations can also be generated as described in the notebook or code files.

9. **Experiment and Fine-Tune**:
   - Feel free to experiment with hyperparameters (like learning rate, batch size, epochs) and model architecture to improve the results.

**Note**: Ensure that your Python environment has the necessary dependencies installed and that the dataset is properly set up before running the scripts.




