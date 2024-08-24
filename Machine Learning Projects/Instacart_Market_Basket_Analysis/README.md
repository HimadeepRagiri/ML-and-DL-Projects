# Instacart Market Basket Analysis with Collaborative Filtering

## Project Overview

This project focuses on performing market basket analysis on the Instacart dataset. The goal is to recommend products to users based on their purchasing history using collaborative filtering. The project includes data exploration, data cleaning, visualization, creation of a user-item matrix, dimensionality reduction using TruncatedSVD, and generating product recommendations using cosine similarity.

## Dataset

The dataset used in this project is the [Instacart Market Basket Analysis dataset](https://www.kaggle.com/c/instacart-market-basket-analysis) available on Kaggle. The dataset contains user orders, product information, and the relationships between users and the products they purchase.

### Key Files:
- **orders.csv**: Contains information about each order, including order ID, user ID, and order sequence.
- **order_products__prior.csv**: Contains detailed product information for each order.
- **products.csv**: Contains product information including product ID and product name.

## Project Workflow

### 1. Importing Necessary Libraries
- **Objective**: Import essential libraries required for data manipulation, visualization, and model building.
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `sklearn`
  - `scipy`

### 2. Loading the Necessary Datasets
- **Objective**: Load the Instacart dataset into Pandas DataFrames.
- **Steps**:
  - Load `orders.csv`, `order_products__prior.csv`, and `products.csv` files.

### 3. Data Exploration
- **Objective**: Understand the structure of the datasets and identify any potential issues.
- **Steps**:
  - Display the first few rows of each dataset.
  - Check for missing values in the datasets.

### 4. Data Cleaning
- **Objective**: Prepare the data for analysis by merging and filtering relevant columns.
- **Steps**:
  - Merge the datasets to create a comprehensive dataset containing user IDs, product IDs, and product names.
  - Filter necessary columns for analysis.
  - Remove duplicate entries to ensure data quality.

### 5. Data Visualization
- **Objective**: Visualize key aspects of the dataset to gain insights.
- **Steps**:
  - Plot the distribution of the number of products purchased per user.
  - Visualize the top 20 most frequently purchased products.

### 6. Create User-Item Matrix
- **Objective**: Build a user-item interaction matrix for collaborative filtering.
- **Steps**:
  - Create a user-item matrix where rows represent users and columns represent products.
  - Use the interaction (1 or 0) to indicate whether a user has purchased a product.

### 7. Apply TruncatedSVD for Dimensionality Reduction
- **Objective**: Reduce the dimensionality of the user-item matrix for better performance in similarity computation.
- **Steps**:
  - Apply TruncatedSVD to reduce the matrix dimensions.
  - Check the explained variance ratio to ensure sufficient variance is captured.

### 8. Compute Similarity and Make Recommendations
- **Objective**: Generate product recommendations based on cosine similarity.
- **Steps**:
  - Compute cosine similarity between items (products) using the reduced-dimension matrix.
  - Create a function to recommend similar products based on a given product ID.

## Example Usage

- **Product Recommendation**: The project includes a function `recommend_products` that takes a `product_id` and the number of recommendations as input and returns a list of similar products.
- **Example**:
  ```python
  recommended_products = recommend_products(1)
  print("Recommended Products:")
  print(recommended_products)