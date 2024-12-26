import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_filename = f"{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pt"
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved at {checkpoint_filename}")


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def visualize_data(data):
    plt.figure(figsize=(10, 5))
    data["rating"].value_counts().sort_index().plot(kind="bar", color="skyblue")
    plt.title("Rating Distribution")
    plt.xlabel("Ratings")
    plt.ylabel("Count")
    plt.show()


def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, checkpoint_path, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_id, movie_id, rating in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            user_id, movie_id, rating = user_id.to(device), movie_id.to(device), rating.to(device)
            optimizer.zero_grad()
            predictions = model(user_id, movie_id)
            loss = criterion(predictions, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for user_id, movie_id, rating in val_loader:
            user_id, movie_id, rating = user_id.to(device), movie_id.to(device), rating.to(device)

            predictions = model(user_id, movie_id)
            loss = criterion(predictions, rating)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for user_id, movie_id, rating in test_loader:
            user_id, movie_id, rating = user_id.to(device), movie_id.to(device), rating.to(device)

            predictions = model(user_id, movie_id)
            loss = criterion(predictions, rating)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss


def recommend_movies(model, user_id, movie_encoder, device, top_n=10, movies_df=None):
    """
    Recommend movies for a given user based on predicted ratings.
    """
    model.eval()

    # Generate predictions for all movies for the given user
    movie_ids = torch.arange(0, len(movie_encoder.classes_)).to(device)
    user_ids = torch.full_like(movie_ids, user_id, dtype=torch.long).to(device)

    # Get predictions for all movies for this user
    with torch.no_grad():
        predicted_ratings = model(user_ids, movie_ids).cpu().numpy()

    # Get the top N movie indices based on predicted ratings
    top_movie_indices = predicted_ratings.argsort()[-top_n:][::-1]

    # Get the movie ids from the movie_encoder
    top_movie_ids = movie_encoder.inverse_transform(top_movie_indices)

    # Retrieve movie titles from the movies dataframe using movie_ids
    recommended_movie_titles = []
    for movie_id in top_movie_ids:
        movie_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values[0]
        recommended_movie_titles.append(movie_title)

    return recommended_movie_titles