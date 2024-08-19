# UNSW-NB15 Network Intrusion Detection

## Project Overview

This project focuses on building a Network Intrusion Detection System (NIDS) using the UNSW-NB15 dataset. The goal is to classify network traffic as either normal or anomalous using a Random Forest Classifier. The project involves various stages, including data exploration, visualization, cleaning, feature selection, model training, and evaluation.

## Dataset

The dataset used in this project is the [UNSW-NB15 dataset](https://www.kaggle.com/mrwellsdavid/unsw-nb15) which consists of network traffic data labeled as either normal or anomalous.

### Key Features:
- **Attributes**: The dataset contains various network traffic attributes such as protocol, service, source IP, destination IP, and other features.
- **Label**: The target variable indicating whether the traffic is `normal` or `anomalous`.

## Project Workflow

### 1. Importing Necessary Libraries
- **Objective**: Import the libraries required for data manipulation, visualization, model training, and evaluation.
- **Libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `sklearn`

### 2. Loading the Data
- **Objective**: Load the training and testing datasets into Pandas DataFrames.
- **Steps**:
  - Load both training and testing datasets.
  - Concatenate them into a single DataFrame for a unified exploration and cleaning process.

### 3. Data Exploration
- **Objective**: Understand the structure, distribution, and potential issues in the dataset.
- **Steps**:
  - Check the shape of the dataset.
  - Generate summary statistics.
  - Check for missing values and data types.

### 4. Data Visualization
- **Objective**: Visualize key aspects of the data for better understanding.
- **Steps**:
  - Plot the distribution of the target variable (normal vs. anomalous traffic).
  - Create a correlation matrix to explore relationships between numeric features.

### 5. Data Cleaning
- **Objective**: Prepare the data for model training by addressing any issues.
- **Steps**:
  - Drop irrelevant columns like `id`.
  - Remove duplicate rows.
  - Convert categorical variables to numerical format using one-hot encoding.

### 6. Feature Selection
- **Objective**: Select relevant features for model training.
- **Steps**:
  - Separate the target variable (`label`) from the feature set.
  - Drop the `label` column from the feature set.

### 7. Data Splitting
- **Objective**: Split the data into training and testing sets for model evaluation.
- **Steps**:
  - Use an 80-20 split to create training and testing datasets.

### 8. Normalization/Standardization
- **Objective**: Standardize the feature set to ensure consistent model performance.
- **Steps**:
  - Apply `StandardScaler` to scale the feature set for both training and testing data.

### 9. Model Training
- **Objective**: Train the Random Forest Classifier on the training data.
- **Steps**:
  - Initialize the `RandomForestClassifier`.
  - Train the model using the standardized training data.

### 10. Model Evaluation
- **Objective**: Evaluate the model's performance on the test set.
- **Steps**:
  - Make predictions on the test data.
  - Assess model performance using accuracy, classification reports, and confusion matrix.

## Results

- **Random Forest Classifier**:
  - Achieved significant accuracy in classifying network traffic as normal or anomalous.
  - The confusion matrix provided insights into the number of true and false positives/negatives.
  - The classification report offered detailed metrics such as precision, recall, and F1-score.

## Conclusion

This project demonstrates the use of Random Forest Classifier in building an effective Network Intrusion Detection System. By preprocessing the data, selecting relevant features, and tuning the model, the classifier was able to distinguish between normal and anomalous traffic with high accuracy.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

## How to Use

1. Clone the repository and navigate to the project directory.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Python script or Jupyter notebook to train the model and evaluate the results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The dataset used in this project was provided by [UNSW](https://www.unsw.edu.au/).
- The inspiration for this project came from various online tutorials and the scikit-learn documentation.
