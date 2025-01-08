import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 64
N_EPOCHS = 5

# File paths
TRAIN_FILE = "NLP_Projects/Named_Entity_Recognition_with_BiLSTM/data/train.csv"
VAL_FILE = "NLP_Projects/Named_Entity_Recognition_with_BiLSTM/data/val.csv"
TEST_FILE = "NLP_Projects/Named_Entity_Recognition_with_BiLSTM/data/test.csv"
CHECKPOINT_DIR = "NLP_Projects/Named_Entity_Recognition_with_BiLSTM/checkpoints"

