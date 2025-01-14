import torch

# =========================================
# Utility Functions for Testing
# =========================================

def create_pad_mask(seq, pad_idx=0):
    """Create mask to ignore padding tokens."""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    """Create a mask for preventing attending to future tokens."""
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.uint8)
    return mask == 0
