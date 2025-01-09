import os
import torch


class Config:
    # Get the base directory of the project
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Training parameters
    BATCH_SIZE = 32
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    MAX_LEN = 50

    # Paths configuration
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")  # Place Flickr8k images here
    CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")  # Place captions file here
    PRETRAINED_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "glove.6B.300d.txt")  # Place GloVe embeddings here
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")  # Place test images here

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        # Create necessary directories if they don't exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.TEST_IMAGES_DIR, exist_ok=True)

        # Verify required files exist
        self._verify_files()

    def _verify_files(self):
        required_files = [
            (self.PRETRAINED_EMBEDDINGS_PATH,
             "GloVe embeddings file not found. Please download from https://nlp.stanford.edu/data/glove.6B.zip"),
            (self.CAPTIONS_FILE, "Captions file not found. Please place your captions.txt in the data directory"),
        ]

        for file_path, message in required_files:
            if not os.path.exists(file_path):
                print(f"WARNING: {message}")
                print(f"Expected path: {file_path}")

