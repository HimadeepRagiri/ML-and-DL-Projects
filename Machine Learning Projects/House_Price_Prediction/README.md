## Housing Price Prediction with Linear Regression and Random Forest

* **Libraries Used:** pandas, numpy, matplotlib, seaborn, sklearn

**# Key Files:**

* `housing.csv`: The dataset containing housing data.
* `House_Price_Prediction.ipynb`: The Python script for data preparation, model training, and evaluation.

**# Steps to Run:**

1. Clone this repository.
2. Install required libraries: `pip install pandas numpy matplotlib seaborn sklearn`
3. Run the script: `House_Price_Prediction.ipynb`

**# Documentation:**

## Data Loading and Exploration

- Loads the dataset using pandas.
- Displays summary information and checks for missing values.
- Handles missing values (e.g., imputation or dropping).

## Data Preprocessing

- Splits data into training and testing sets.
- Log-transforms features with skewed distributions (if necessary).
- Performs one-hot encoding for categorical features.
- Creates additional features through feature engineering.
- Normalizes features using StandardScaler (optional).

## Model Building and Training

### Linear Regression Model

- Creates a LinearRegression model.
- Trains the model on the preprocessed training data.

### Random Forest Regressor Model

- Creates a RandomForestRegressor model.
- Trains the model on the preprocessed training data.

## Model Evaluation

- Evaluates both models using metrics like R-squared, mean squared error (MSE), or mean absolute error (MAE) on the testing set.
- Compares performance and chooses the best model based on your evaluation criteria.

## Visualizations (Optional)

- Generates histograms to visualize feature distributions.
- Creates a correlation heatmap to explore feature relationships.
- Plots a scatterplot of latitude and longitude colored by median house value.

## Contributions

Feel free to contribute to this project by suggesting improvements, reporting issues, or creating pull requests.
