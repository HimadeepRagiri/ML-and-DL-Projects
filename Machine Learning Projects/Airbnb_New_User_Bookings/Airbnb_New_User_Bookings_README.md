# Airbnb New User Bookings Prediction

This project involves predicting the destination countries of new Airbnb users based on their initial data.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- xgboost

## Key Files
- `airbnb_new_user_bookings.csv`: The dataset containing user and booking information.
- `Airbnb_New_User_Bookings_Prediction.ipynb`: The Jupyter Notebook for data preparation, model training, and evaluation.

## Steps to Run
1. Clone this repository.
2. Install the required libraries:
3. Open and run the notebook: `Airbnb_New_User_Bookings_Prediction.ipynb`.

## Documentation

### Data Loading and Exploration
- Loads the dataset using pandas.
- Displays summary information and checks for missing values.
- Handles missing values by imputing mean for numerical features and 'unknown' for categorical features.

### Data Preprocessing
- Splits data into training and testing sets.
- Extracts features such as year, month, and day from `timestamp_first_active`.
- Encodes categorical variables using OneHotEncoder.
- Normalizes numerical features using StandardScaler.

### Model Building and Training

#### Logistic Regression Model
- Creates a `LogisticRegression` model.
- Trains the model on the preprocessed training data.

#### XGBoost Classifier Model
- Creates an `XGBClassifier` model.
- Trains the model on the preprocessed training data.

### Model Evaluation
- Evaluates both models using metrics like accuracy, confusion matrix, and classification report on the testing set.
- Compares performance and determines the best model based on evaluation metrics.

### Visualizations (Optional)
- Generates histograms to visualize the distribution of features like age.
- Creates count plots to visualize the distribution of categorical variables like country destination.

## Contributions
Feel free to contribute to this project by suggesting improvements, reporting issues, or creating pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
