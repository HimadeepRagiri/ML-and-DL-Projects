# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wxg0bMtukPnyE9X9RfCSBu1dEDOAewVF
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Subset, DataLoader, TensorDataset

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """ Save model weights, optimizer state, and scheduler state """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
    """ Load the checkpoint and resume training """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filename}")
    return model, optimizer, scheduler, epoch, loss

def test_model(model, loader, criterion, device):
    """ Evaluate the model and calculate metrics """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
            y_batch = y_batch.unsqueeze(1)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item()

            preds = torch.round(torch.sigmoid(y_pred))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {running_loss / len(loader):.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return accuracy, precision, recall, f1, cm

def create_subset_loader(dataset, subset_fraction, batch_size, shuffle=True):
    """ Create a subset loader for a smaller dataset """
    total_size = len(dataset)
    subset_size = int(subset_fraction * total_size)
    subset_indices = torch.randperm(total_size)[:subset_size].tolist()
    subset = Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

def prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float16)
    y_train_tensor = torch.tensor(y_train.values.ravel(), dtype=torch.float16)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float16)
    y_test_tensor = torch.tensor(y_test.values.ravel(), dtype=torch.float16)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_train_tensor

def test_learning_rates(model_class, input_size, num_classes, device, small_train_loader, learning_rates, num_epochs=3):
    """
    Tests different learning rates to find the best one based on loss.

    Args:
        model_class (torch.nn.Module): The model class (e.g., CreditRiskNN).
        input_size (int): The number of input features.
        num_classes (int): The number of output classes.
        device (torch.device): The device (CPU or GPU) for training.
        small_train_loader (DataLoader): The training DataLoader with a smaller dataset for quick evaluation.
        learning_rates (list): A list of learning rates to test.
        num_epochs (int): The number of epochs for training.

    Returns:
        tuple: (best_lr, best_loss) - The best learning rate and corresponding loss.
    """
    best_lr = None
    best_loss = float('inf')

    for lr in learning_rates:
        model = model_class(input_size=input_size, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Train the model with the smaller dataset (quick evaluation)
        model.train()
        running_loss = 0.0

        for epoch in range(num_epochs):  # Train for a few epochs only to test
            for X_batch, y_batch in small_train_loader:
                X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
                y_batch = y_batch.unsqueeze(1)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_loss = running_loss / len(small_train_loader)
        print(f"Learning rate: {lr}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lr = lr

    return best_lr, best_loss