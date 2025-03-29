from transformers import BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve your token securely from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Enable GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization config
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
