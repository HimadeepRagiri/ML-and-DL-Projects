from data_utils import load_and_analyze_data, preprocess_function, prepare_dataloader, analyze_dataset
from model_utils import setup_model, prepare_data_for_training
from train_eval import setup_trainer, train_model, evaluate_model
from visualize import plot_training_loss
from inference import generate_headline

def main():
    # Load and analyze data
    dataset, train_sample = load_and_analyze_data()

    # Preprocessing
    print("\nApplying preprocessing to the training sample...")
    train_sample = train_sample.map(preprocess_function)
    print("\nPost-Preprocessing Sample:")
    print("Article snippet:", train_sample[0]['article'][:200], "...")
    print("Headline:", train_sample[0]['highlights'])

    # DataLoader check
    train_dataloader = prepare_dataloader(train_sample)
    batch = next(iter(train_dataloader))
    print("\nDataLoader Batch Sample:")
    print("Articles (first 2):", batch["article"][:2])
    print("Headlines (first 2):", batch["headline"][:2])

    # Model setup
    model, tokenizer = setup_model()

    # Prepare tokenized datasets
    val_sample = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
    val_sample = val_sample.map(preprocess_function)
    train_tokenized, val_tokenized, data_collator = prepare_data_for_training(train_sample, val_sample, tokenizer)

    # Training
    trainer = setup_trainer(model, train_tokenized, val_tokenized, data_collator, tokenizer)
    trainer = train_model(trainer)

    # Visualization
    plot_training_loss(trainer)

    # Evaluation
    evaluate_model(trainer)

    # Inference test
    sample_article = val_sample[0]['article']
    print("\nSample Article (first 300 characters):")
    print(sample_article[:300])
    print("\nGenerated Headline:")
    print(generate_headline(sample_article, model, tokenizer))

if __name__ == "__main__":
    main()
    