import os

def setup_directories(config):
    """Create necessary directories for saving artifacts."""
    directories = [
        config.base_dir,
        config.checkpoints_dir,
        config.tensorboard_dir,
        config.model_dir,
        config.logs_dir
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def inspect_training_examples(train_dataset, tokenized_train, tokenizer, config, num_examples=5):
    """
    Inspect training examples to verify data processing and tokenization.

    Args:
        train_dataset: Raw training dataset with code and documentation
        tokenized_train: Tokenized training dataset
        tokenizer: The tokenizer instance
        config: Configuration instance
        num_examples: Number of examples to display (default=5)
    """
    print(f"\n{'='*80}")
    print("TRAINING EXAMPLES INSPECTION")
    print(f"{'='*80}")

    # Get random indices and convert them to Python ints
    indices = [int(x) for x in np.random.randint(0, len(train_dataset), num_examples)]

    for idx in indices:
        print(f"\nExample {idx+1}")
        print(f"{'-'*80}")

        # Original data
        print("ORIGINAL DATA:")
        print("\nCode:")
        print(train_dataset[idx]['code'])
        print("\nDocumentation:")
        print(train_dataset[idx]['documentation'])

        # Formatted data (how it looks after special tokens)
        formatted_text = (f"{config.code_token}{train_dataset[idx]['code']}"
                          f"{config.sep_token}{config.doc_token}"
                          f"{train_dataset[idx]['documentation']}")
        print("\nFORMATTED TEXT:")
        print(formatted_text)

        # Tokenized data
        print("\nTOKENIZED AND DECODED:")
        tokenized_text = tokenizer.decode(tokenized_train[idx]['input_ids'])
        print(tokenized_text)

        # Token statistics
        num_tokens = len([t for t in tokenized_train[idx]['input_ids'] if t != tokenizer.pad_token_id])
        print(f"\nNumber of tokens (excluding padding): {num_tokens}")
        print(f"Maximum sequence length: {config.max_length}")

        print(f"\n{'='*80}")

def save_model_and_tokenizer(model, tokenizer, config):
    """Save the final model and tokenizer."""
    print(f"\nSaving model and tokenizer to {config.model_dir}")
    model.save_pretrained(config.model_dir)
    tokenizer.save_pretrained(config.model_dir)
