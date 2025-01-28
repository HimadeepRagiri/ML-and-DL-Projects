import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Creates a synthetic dataset with simple patterns for testing the GPT model.
    Each sequence follows a pattern like: [1, 2, 3, 4] -> [2, 3, 4, 5]
    """

    def __init__(self, num_samples, seq_length, vocab_size):
        self.data = []
        for _ in range(num_samples):
            # Create sequences with simple patterns
            start = np.random.randint(0, vocab_size - seq_length)
            seq = np.arange(start, start + seq_length) % vocab_size
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]