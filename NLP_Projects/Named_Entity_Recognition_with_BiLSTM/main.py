import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
import spacy
from src.config import *
from src.data_preprocessing import (
    CustomNERDataset,
    parse_conll_file,
    save_to_csv,
    build_vocab,
    create_data_loaders
)
from src.model import BiLSTMNER
from src.train import train_model, evaluate
from src.utils import predict
from app import start_server

def main():
    # Initialize spacy and tokenizer
    spacy.load("en_core_web_sm")
    tokenizer = get_tokenizer("basic_english")

    # Parse and save data
    train_sentences, train_tags = parse_conll_file("NLP_Projects/Named_Entity_Recognition_with_BiLSTM/data/conll2003/eng.train")
    val_sentences, val_tags = parse_conll_file("NLP_Projects/Named_Entity_Recognition_with_BiLSTM/data/conll2003/eng.testb")
    test_sentences, test_tags = parse_conll_file("NLP_Projects/Named_Entity_Recognition_with_BiLSTM/data/conll2003/eng.testa")

    save_to_csv(train_sentences, train_tags, TRAIN_FILE)
    save_to_csv(val_sentences, val_tags, VAL_FILE)
    save_to_csv(test_sentences, test_tags, TEST_FILE)

    # Create datasets
    train_dataset = CustomNERDataset(TRAIN_FILE, tokenizer)
    val_dataset = CustomNERDataset(VAL_FILE, tokenizer)
    test_dataset = CustomNERDataset(TEST_FILE, tokenizer)

    # Build vocabularies
    TEXT, TAGS = build_vocab(train_dataset)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        TEXT, TAGS, BATCH_SIZE
    )

    # Initialize model
    model = BiLSTMNER(
        input_dim=len(TEXT),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(TAGS),
        pad_idx=TEXT.get_stoi()['<pad>']
    ).to(DEVICE)

    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=TAGS.get_stoi()['<pad>'])

    # Train the model
    train_model(
        model, train_loader, val_loader,
        optimizer, criterion, DEVICE,
        N_EPOCHS, CHECKPOINT_DIR
    )

    # Final evaluation
    print("\nEvaluating test set...")
    test_loss = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")

    # Example prediction
    example = "John works at Microsoft in New York."
    print("\nPredicting NER tags for:", example)
    results = predict(model, example, tokenizer, TEXT, TAGS, DEVICE)

    # start the web server
    print("\nStarting web server...")
    start_server(model, TEXT, TAGS)

if __name__ == "__main__":
    main()