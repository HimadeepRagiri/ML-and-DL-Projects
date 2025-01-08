import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils import save_checkpoint

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        tokens, tags = batch
        tokens = tokens.to(device)
        tags = tags.to(device)

        optimizer.zero_grad()
        predictions = model(tokens)

        batch_size, seq_len, num_classes = predictions.shape
        predictions = predictions.view(-1, num_classes)
        tags = tags.view(-1)

        assert predictions.shape[0] == tags.shape[0], \
            f"Prediction shape {predictions.shape} doesn't match target shape {tags.shape}"

        loss = criterion(predictions, tags)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

    return epoch_loss / len(train_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            tokens, tags = batch
            tokens = tokens.to(device)
            tags = tags.to(device)

            predictions = model(tokens)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def train_model(model, train_loader, val_loader, optimizer, criterion, device, n_epochs, checkpoint_dir):
    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}")

        save_checkpoint(
            model,
            optimizer,
            epoch,
            avg_loss,
            checkpoint_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pth"
        )

        # Validate after each epoch
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")