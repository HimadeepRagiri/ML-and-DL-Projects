# Spam Email Classification with Logistic Regression

This repository contains Python code for classifying spam and non-spam emails using Logistic Regression.

## Dependencies

* numpy
* pandas
* sklearn (scikit-learn)

## Data

The code assumes you have a CSV file named `mail_data.csv` containing two columns:

* `Message`: The text content of the email.
* `Category`: The category of the email (e.g., "spam", "ham").

## Steps

1. **Import libraries:** Necessary libraries for data manipulation, model building, and evaluation are imported.
2. **Load data:** The `mail_data.csv` file is loaded into a pandas DataFrame.
3. **Preprocess data:**
    * Null values are replaced with an empty string.
    * Data is explored using `head()` and `shape` methods.
4. **Feature extraction:**
    * Features (email text) are separated from target labels (category).
    * The data is split into training and testing sets using `train_test_split`.
    * A `TfidfVectorizer` is used to transform text data into numerical features:
        * `min_df=1`: Ensures that each word appears at least once in the training data.
        * `stop_words='english'`: Removes common stop words like "the", "a", etc.
        * `lowercase=True`: Converts all text to lowercase.
    * Training and testing features are created using the fitted vectorizer.
5. **Label encoding:**
    * Target labels (`y_train` and `y_test`) are converted to integers for compatibility with the model.
6. **Model creation:**
    * A Logistic Regression model is created.
7. **Model training:**
    * The model is trained on the training features and labels.
8. **Model evaluation:**
    * The model's accuracy on both training and testing data is calculated using the `accuracy_score` function.

## Running the code

1. Save the code as a Python file (e.g., `Spam_Mail_Detection.ipynb`).
2. Ensure you have the required libraries installed (`pip install numpy pandas sklearn`).
3. Place your `mail_data.csv` file in the same directory as the Python script.
4. Run the script from the command line: `Spam_Mail_Detection.ipynb`

## Results

The script will print the training and testing accuracy of the model.

## Further exploration

* Experiment with different parameters of the `TfidfVectorizer` (e.g., `max_features`, `ngram_range`).
* Try different machine learning models (e.g., Naive Bayes, Support Vector Machines).
* Explore other evaluation metrics (e.g., precision, recall, F1-score).