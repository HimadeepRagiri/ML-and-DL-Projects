from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import HF_TOKEN, MODEL_NAME, BNB_CONFIG, DEVICE
import torch

#Declare globals
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    return model, tokenizer

# Function to call the LLM
def call_llm(prompt: str, max_tokens: int = 500) -> str:
    global model, tokenizer
    # Construct a prompt that strongly emphasizes JSON response format
    full_prompt = (
        f"[INST] {prompt}\n\n"
        f"IMPORTANT: Your response must be a valid JSON object only. Format your response as proper JSON.\n"
        f"DO NOT include any text outside the JSON object.\n"
        f"Example of correct format: {{\"key\": \"value\", \"otherKey\": 123}}\n"
        f"Do not use markdown formatting for JSON. [/INST]"
    )

    # Tokenize and generate response
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,  # Lower temperature for more predictable outputs
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part after [/INST]
    if "[/INST]" in response:
        response = response.split("[/INST]", 1)[1].strip()

    return response
