import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch


def load_data(ratings_path, movies_path):
    # Load ratings.dat with specified encoding
    ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"], encoding='ISO-8859-1')

    # Load movies.dat with specified encoding
    movies = pd.read_csv(movies_path, sep="::", engine="python",
                         names=["movie_id", "title", "genres"], encoding='ISO-8859-1')

    return ratings, movies


def preprocess_data(ratings):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    # Encode user and movie IDs
    ratings["user_id"] = user_encoder.fit_transform(ratings["user_id"])
    ratings["movie_id"] = movie_encoder.fit_transform(ratings["movie_id"])

    # Split into train, validation, and test sets
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data, user_encoder, movie_encoder


class MovieDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.data.iloc[idx]["user_id"]
        movie_id = self.data.iloc[idx]["movie_id"]
        rating = self.data.iloc[idx]["rating"]
        return torch.tensor(user_id, dtype=torch.long), \
            torch.tensor(movie_id, dtype=torch.long), \
            torch.tensor(rating, dtype=torch.float)


def create_dataloaders(train, val, test, batch_size=512):
    train_dataset = MovieDataset(train)
    val_dataset = MovieDataset(val)
    test_dataset = MovieDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader