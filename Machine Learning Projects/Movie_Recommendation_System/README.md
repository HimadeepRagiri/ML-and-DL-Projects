# Movie Recommendation System

This project is a content-based movie recommendation system built using Python. It uses the TF-IDF (Term Frequency-Inverse Document Frequency) technique and cosine similarity to recommend movies similar to the user's input movie.

## Dataset

The dataset used for this project is 'movies.csv', which contains information about movies, including their titles, genres, keywords, taglines, cast, and directors. This dataset is not included in the repository and needs to be obtained separately.

## Requirements

To run this project, you'll need the following libraries installed:

- NumPy
- Pandas
- difflib (from Python's standard library)
- scikit-learn (sklearn)

You can install the required libraries using pip:

## Usage

1. Clone the repository or download the source code.
2. Place the 'movies.csv' dataset in the same directory as the Python script.
3. Run the Python script.
4. When prompted, enter the name of your favorite movie.
5. The script will recommend up to 10 similar movies based on the content similarity.

## How it Works

1. The script loads the 'movies.csv' dataset into a Pandas DataFrame.
2. Relevant features (genres, keywords, tagline, cast, and director) are selected for recommendation.
3. The selected features are combined into a single string for each movie.
4. The combined features are converted into feature vectors using TF-IDF vectorization.
5. Cosine similarity between the feature vectors is calculated to find the similarity between movies.
6. The user is asked to enter their favorite movie name.
7. The closest match to the entered movie name is found using difflib's `get_close_matches` function.
8. The index of the closest match movie is determined.
9. The similarity scores between the closest match movie and all other movies are calculated.
10. The movies are sorted based on their similarity scores in descending order.
11. The top 10 most similar movies are recommended to the user.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.