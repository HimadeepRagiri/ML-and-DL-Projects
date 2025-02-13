from transformers import GPT2Tokenizer

def initialize_tokenizer(config):
    """Initialize and configure the tokenizer with special tokens."""
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)

    # Add special tokens for documentation generation
    special_tokens = {
        'additional_special_tokens': [
            config.code_token,
            config.doc_token,
            config.sep_token
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
