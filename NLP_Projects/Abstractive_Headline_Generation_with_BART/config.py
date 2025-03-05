class Config:
    """
    Configuration class containing all tunable hyperparameters and settings.
    """
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    SAMPLE_SIZE = 10000         # Use a subset (10K samples)
    MAX_INPUT_LENGTH = 512      # Maximum token length for articles
    MAX_TARGET_LENGTH = 64      # Maximum token length for headlines
    BATCH_SIZE = 8              # Batch size for DataLoader
    NUM_WORKERS = 2             # Number of workers for DataLoader
    LEARNING_RATE = 3e-4        # Learning Rate
    TRAIN_EPOCHS = 3            # Number of Epochs for training

    # paths
    output_dir = './results'

# Instantiate the configuration
config = Config()
