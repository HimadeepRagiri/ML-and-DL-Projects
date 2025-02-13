import torch
from config.config import Config
from src.data.data_processing import load_and_analyze_data, preprocess_datasets
from src.model.model import initialize_model, generate_documentation
from src.model.tokenizer import initialize_tokenizer
from src.training.trainer import setup_training, train_model, evaluate_model
from src.utils.utils import setup_directories, inspect_training_examples, save_model_and_tokenizer
from torch.utils.tensorboard import SummaryWriter


def main():
    # Initialize configuration
    config = Config()

    # Setup directories
    setup_directories(config)

    # Load and analyze data
    train_dataset, val_dataset = load_and_analyze_data(config)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(config)

    # Preprocess datasets
    tokenized_train, tokenized_val = preprocess_datasets(
        train_dataset, val_dataset, tokenizer, config
    )

    # Inspect training examples
    inspect_training_examples(train_dataset, tokenized_train, tokenizer, config)

    # Initialize model
    model = initialize_model(config, tokenizer)

    # Setup training
    trainer = setup_training(
        config, model, tokenized_train, tokenized_val, tokenizer
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    # Train model
    train_result = train_model(trainer, config)

    # Evaluate model
    eval_results = evaluate_model(trainer, writer)

    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, config)

    # Close TensorBoard writer
    writer.close()

    # Example documentation generation
    code_example = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    documentation = generate_documentation(code_example, model, tokenizer)
    print(f"\nExample Documentation Generation:")
    print(f"Input Code:\n{code_example}")
    print(f"Generated Documentation:\n{documentation}")

if __name__ == "__main__":
    main()