from transformers import BertForSequenceClassification
from config import Config

def initialize_model():
    """Initialize the BERT model for sequence classification"""
    return BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS
    )