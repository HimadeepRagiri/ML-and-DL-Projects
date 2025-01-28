import torch

class Hyperparameters:
    def __init__(self):
        self.vocab_size = 100        # Smaller vocabulary for synthetic data
        self.embed_size = 256        # Embedding dimension
        self.num_layers = 3          # Number of transformer layers
        self.heads = 4               # Number of attention heads
        self.forward_expansion = 4    # FFN expansion factor
        self.dropout = 0.1           # Dropout rate
        self.max_length = 20         # Maximum sequence length
        self.batch_size = 32         # Batch size
        self.epochs = 10             # Number of training epochs
        self.lr = 3e-4               # Learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")