import re
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from config import Config

# Data Loading & Analysis
def load_and_analyze_data():
    print("Loading dataset...")
    dataset = load_dataset(Config.DATASET_NAME, Config.DATASET_VERSION)

    # Print dataset splits and sizes
    print("Dataset splits and sizes:")
    for split in dataset.keys():
        print(f"{split} split has {len(dataset[split])} samples.")

    # For analysis, work with a subset of the training data
    train_sample = dataset['train'].select(range(min(Config.SAMPLE_SIZE, len(dataset['train']))))

    analyze_dataset(train_sample, "Train Sample")
    return dataset, train_sample

def analyze_dataset(split_data, split_name="Train"):
    """
    Analyze dataset by calculating the average number of words in articles and headlines,
    and plotting their distributions.
    """
    article_lengths = [len(item['article'].split()) for item in split_data]
    headline_lengths = [len(item['highlights'].split()) for item in split_data]

    print(f"\n{split_name} Data Analysis:")
    print(f"Number of samples: {len(split_data)}")
    print(f"Average article length (words): {np.mean(article_lengths):.2f}")
    print(f"Average headline length (words): {np.mean(headline_lengths):.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(article_lengths, bins=50, color='skyblue')
    plt.title(f"{split_name} Article Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(headline_lengths, bins=50, color='salmon')
    plt.title(f"{split_name} Headline Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Preprocessing
def clean_text(text):
    """
    Basic text cleaning function:
    - Strips whitespace from the ends.
    - Replaces multiple spaces/newlines with a single space.
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_function(example):
    """
    Preprocess a single example by cleaning both the article and the headline.
    """
    example['article'] = clean_text(example['article'])
    example['highlights'] = clean_text(example['highlights'])
    return example

# Dataset/Dataloader
class CNNDailyMailDataset(Dataset):
    """
    Custom PyTorch Dataset to wrap the preprocessed CNN/DailyMail data.
    """
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {"article": item['article'], "headline": item['highlights']}

def prepare_dataloader(train_sample):
    train_data = CNNDailyMailDataset(train_sample)
    train_dataloader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    return train_dataloader
