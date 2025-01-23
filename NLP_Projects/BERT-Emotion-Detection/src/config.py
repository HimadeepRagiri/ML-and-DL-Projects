class Config:
    """Configuration settings for the emotion classification project"""

    # Dataset
    DATASET_NAME = "emotion"
    TEXT_FIELD = "text"
    LABEL_FIELD = "label"
    LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # Model
    MODEL_NAME = "bert-base-uncased"
    NUM_LABELS = 6

    # Training
    MAX_LENGTH = 64
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01

    # Paths
    CHECKPOINT_DIR = "/HimadeepRagiri/ML-and-DL-Projects/NLP_Projects/BERT-Emotion-Detection/checkpoints"
    LOG_DIR = "/HimadeepRagiri/ML-and-DL-Projects/NLP_Projects/BERT-Emotion-Detection/logs"