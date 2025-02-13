import torch
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

def initialize_model(config, tokenizer):
    """Initialize and configure the model with LoRA."""
    print("Initializing model...")
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def generate_documentation(code, model, tokenizer, config, max_length=200):
    """Generate documentation for a given code snippet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Format input with special tokens
    prompt = f"{config.code_token}{code}{config.sep_token}{config.doc_token}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate documentation
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the documentation part
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    doc_start = generated_text.find(config.doc_token) + len(config.doc_token)
    documentation = generated_text[doc_start:].strip()

    return documentation
