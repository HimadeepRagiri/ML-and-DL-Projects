import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import TavilySearchResults

# Load Quantized LLM (4-bit Mistral)
model_id = "mistralai/Mistral-7B-v0.1"
quant_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto"
)

# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Web Search Tool
search_tool = TavilySearchResults()

def load_models():
    return tokenizer, model, embedding_model, search_tool
