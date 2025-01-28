import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config.hyperparameters import Hyperparameters
from data.dataset import SyntheticDataset
from models.gpt import GPT
from utils.training import train_model
from utils.generation import make_predictions

def main():
    # Initialize hyperparameters
    hp = Hyperparameters()

    # Create synthetic dataset
    train_dataset = SyntheticDataset(1000, hp.max_length, hp.vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)

    # Initialize model
    model = GPT(
        vocab_size=hp.vocab_size,
        embed_size=hp.embed_size,
        num_layers=hp.num_layers,
        heads=hp.heads,
        forward_expansion=hp.forward_expansion,
        dropout=hp.dropout,
        max_length=hp.max_length
    ).to(hp.device)

    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    print("Training GPT model...")
    losses = train_model(model, train_loader, optimizer, criterion,
                        hp.device, hp.epochs)

    # Plot training losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.show()

    # Make predictions
    make_predictions(model, hp)

if __name__ == "__main__":
    main()