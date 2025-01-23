import os
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config
from collections import Counter


def load_and_preprocess_data():
    """Load and preprocess the dataset, returning dataloaders and class weights"""
    # Load dataset
    dataset = load_dataset(Config.DATASET_NAME)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)

    # Preprocessing function
    def preprocess(batch):
        encoded = tokenizer(
            batch[Config.TEXT_FIELD],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
            return_token_type_ids=False
        )
        return {
            "input_ids": encoded["input_ids"].squeeze().numpy(),
            "attention_mask": encoded["attention_mask"].squeeze().numpy(),
            "labels": batch[Config.LABEL_FIELD]
        }

    # Apply preprocessing
    dataset = dataset.map(preprocess, batched=True, batch_size=32)

    # Create datasets
    class EmotionDataset(Dataset):
        def __init__(self, data):
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            self.labels = data["labels"]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.input_ids[idx]),
                "attention_mask": torch.tensor(self.attention_mask[idx]),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }

    train_dataset = EmotionDataset(dataset["train"])
    val_dataset = EmotionDataset(dataset["validation"])
    test_dataset = EmotionDataset(dataset["test"])

    # Calculate class weights
    label_counts = Counter(dataset["train"][Config.LABEL_FIELD])
    class_counts = torch.tensor([label_counts[i] for i in range(6)])
    class_weights = (1.0 / class_counts) * class_counts.sum() / 2.0

    # Create dataloaders
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch])
        }

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    return dataset, train_loader, val_loader, test_loader, class_weights, tokenizer