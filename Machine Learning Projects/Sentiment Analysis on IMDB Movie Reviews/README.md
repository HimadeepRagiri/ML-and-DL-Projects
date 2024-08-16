# Sentiment Analysis on IMDB Movie Reviews

## Project Overview

This project is focused on classifying IMDB movie reviews as either positive or negative using the Multinomial Naive Bayes algorithm. The project involves several key steps, including data exploration, cleaning, feature engineering, model training, evaluation, and hyperparameter tuning to achieve optimal results.

## Dataset

The dataset used in this project is the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-movie-reviews), which consists of movie reviews and their corresponding sentiment labels (either `positive` or `negative`).

### Key Features:
- **Review**: The text of the movie review.
- **Sentiment**: The label indicating the sentiment of the review (`positive` or `negative`).

## Project Workflow

### 1. Data Exploration
- **Objective**: Understand the structure and distribution of the data.
- **Steps**:
  - Inspect the first few rows of the dataset.
  - Check the shape of the dataset.
  - Analyze the distribution of sentiment labels.
  - Explore basic statistics and check for any missing or duplicated data.

### 2. Data Visualization
- **Objective**: Visualize key characteristics of the data.
- **Steps**:
  - Plot the distribution of sentiment labels.
  - Analyze the distribution of review lengths to understand variability in the data.

### 3. Data Cleaning
- **Objective**: Prepare the data for modeling.
- **Steps**:
  - Remove duplicate rows.
  - Clean the text data by converting it to lowercase, removing numbers, punctuation, and extra spaces.

### 4. Feature Engineering and Selection
- **Objective**: Transform text data into numerical features for model training.
- **Steps**:
  - Use TF-IDF Vectorization to convert the cleaned text into numerical features.
  - Limit the features to the top 5,000 most relevant words for efficiency.

### 5. Standardization/Normalization
- **Objective**: Standardize the feature set for consistent model performance.
- **Steps**:
  - Apply StandardScaler to ensure the features are on a similar scale.

### 6. Train-Test Split
- **Objective**: Split the dataset into training and testing sets.
- **Steps**:
  - Use an 80-20 split to train the model on 80% of the data and test it on the remaining 20%.

### 7. Model Training
- **Objective**: Train a Multinomial Naive Bayes model on the data.
- **Steps**:
  - Fit the model to the training data.
  - Evaluate the model's performance using accuracy, classification reports, and confusion matrix.

### 8. Model Evaluation
- **Objective**: Assess the model's effectiveness on unseen data.
- **Steps**:
  - Predict sentiment labels on the test data.
  - Compare predictions to actual labels using metrics such as accuracy and confusion matrix.

### 9. Hyperparameter Tuning with GridSearchCV
- **Objective**: Optimize the model's performance by tuning hyperparameters.
- **Steps**:
  - Define a grid of potential values for the `alpha` parameter.
  - Use GridSearchCV to find the best combination of hyperparameters through cross-validation.

## Results

- **Best Accuracy**: Achieved after tuning the `alpha` parameter using GridSearchCV.
- **Confusion Matrix**: Provided insight into the number of correct and incorrect predictions.
- **Classification Report**: Offered detailed performance metrics such as precision, recall, and F1-score.

## Conclusion

This project successfully demonstrates the application of the Multinomial Naive Bayes algorithm for sentiment analysis on the IMDB movie reviews dataset. The model was optimized using hyperparameter tuning, resulting in improved accuracy and overall performance. This workflow serves as a strong foundation for tackling similar text classification problems.

## Requirements

To run this project, the following Python libraries are required:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `re`

## How to Use

1. Clone the repository and navigate to the project directory.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook or Python script to see the results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The dataset used in this project was provided by [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-movie-reviews).
- Inspiration for this project came from various online tutorials and the scikit-learn documentation.
