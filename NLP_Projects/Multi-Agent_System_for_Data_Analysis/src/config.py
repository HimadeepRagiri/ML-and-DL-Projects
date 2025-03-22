from transformers import BitsAndBytesConfig
import torch

HF_TOKEN = "Replace with your actual token"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Enable GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization config
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)