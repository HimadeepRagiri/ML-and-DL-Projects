import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import load_data, preprocess_data, create_dataloaders
from model import RecommenderNet
from utils import (
    visualize_data, train_model, validate_model,
    test_model, recommend_movies
)

def main():
    # File paths
    ratings_path = "/path/to/ML-and-DL-Projects/Deep Learning Projects/DeepLearning_Recommendation_System/dataset/ml-1m/ratings.dat"
    movies_path = "/path/to/ML-and-DL-Projects/Deep Learning Projects/DeepLearning_Recommendation_System/dataset/ml-1m/movies.dat"
    checkpoint_path = "/path/to/ML-and-DL-Projects/Deep Learning Projects/DeepLearning_Recommendation_System/checkpoint_epoch_10.pt"

    # Load and preprocess data
    ratings, movies = load_data(ratings_path, movies_path)
    print("Ratings Dataset:\n", ratings.head())
    print("\nMovies Dataset:\n", movies.head())

    # Visualize the ratings distribution
    visualize_data(ratings)

    # Preprocess data
    train_data, val_data, test_data, user_encoder, movie_encoder = preprocess_data(ratings)
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_users = ratings["user_id"].nunique()
    num_movies = ratings["movie_id"].nunique()
    model = RecommenderNet(num_users, num_movies).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Evaluate the model
    validation_loss = validate_model(model, val_loader, criterion, device)
    print(f"Validation Loss: {validation_loss:.4f}")

    test_loss = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Generate recommendations for a sample user
    user_id = 1
    recommended_movies = recommend_movies(
        model=model,
        user_id=user_id,
        movie_encoder=movie_encoder,
        device=device,
        top_n=10,
        movies_df=movies
    )

    print(f"\nRecommended Movies for User {user_id}:")
    for movie in recommended_movies:
        print(movie)

if __name__ == "__main__":
    main()