# Customer Segmentation using K-Means Clustering

This project demonstrates customer segmentation using K-Means clustering algorithm. Customer segmentation is a process of dividing customers into groups based on shared characteristics such as demographics, behavior, or preferences. In this project, we segment customers based on their annual income and spending score.

## Dataset
The dataset used in this project is named "Mall_Customers.csv". It contains information about customers including their gender, age, annual income, and spending score.

## Setup
To run the project, make sure you have the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

## Steps
1. **Loading Data**: Load the dataset into a pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Check the shape of the dataset.
   - Get information about the dataset.
   - Check for missing values.
3. **Choosing Features**: Select the relevant features for clustering (Annual Income and Spending Score).
4. **Finding Optimal Number of Clusters**: Use the Elbow method to determine the optimal number of clusters.
5. **Training the Model**: Train the K-Means clustering model with the optimal number of clusters.
6. **Predicting Clusters**: Assign each data point to a cluster.
7. **Visualizing Clusters**: Plot the clusters and centroids on a scatter plot.

## Usage
1. Clone this repository: https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
2. Install required libraries:
3. Run the script: `Customer_Segmentation.ipynb`

## Results
The project generates a visualization of customer segments based on their annual income and spending score. The clusters are represented by different colors, and centroids are marked with black points.
