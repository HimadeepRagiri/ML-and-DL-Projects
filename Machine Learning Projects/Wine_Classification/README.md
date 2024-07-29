# Wine Classification with K-Nearest Neighbors (KNN)

This project focuses on building a classification model to predict the class of wines based on various features. The dataset used in this project is the Wine Dataset from the scikit-learn library.

## Project Structure

The project consists of the following components:

1. **Data Import and Exploration**: 
   - Importing necessary libraries including NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn.
   - Loading the Wine Dataset from scikit-learn.
   - Exploring the dataset using Pandas DataFrame.

2. **Data Preprocessing**:
   - Checking for null values in the dataset.
   - Splitting the dataset into features (X) and target (y).
   - Splitting the data into training and testing sets using train_test_split from scikit-learn.

3. **Model Development**:
   - Creating a K-Nearest Neighbors (KNN) classifier model.
   - Training the KNN model on the training data.
   - Evaluating the model's performance on both the training and test datasets.

4. **Model Evaluation and Tuning**:
   - Tuning the sensitivity of the model to the number of neighbors (n_neighbors) using a range of values.
   - Visualizing the accuracy of the model for different values of K.
   - Evaluating the effect of changing the training split percentage on model accuracy.

5. **Model Deployment**:
   - Making predictions using the trained model on the test data.
   - Generating a confusion matrix to evaluate the model's performance.

## Usage

To run this project:

1. Open the provided Jupyter Notebook in Google Colab or any Python environment.
2. Run each cell sequentially to execute the code and observe the results.
3. Modify the code or experiment with different parameters to further improve the model's performance.

## Dependencies

This project requires the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
