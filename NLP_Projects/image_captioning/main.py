import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import os

from config import Config
from data_preprocessing import (
    load_glove_embeddings,
    build_vocab,
    FlickrDataset,
)
from model import ImageCaptioningModel
from train import train_model
from utils import test_example


def main():
    # Initialize configuration
    config = Config()

    # Verify all required directories and files exist
    if not os.path.exists(config.IMAGE_DIR) or not os.listdir(config.IMAGE_DIR):
        raise FileNotFoundError(
            f"Image directory is empty or not found at {config.IMAGE_DIR}. "
            "Please place your Flickr8k images in this directory."
        )

    print("Loading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings(config.PRETRAINED_EMBEDDINGS_PATH)

    print("Loading captions dataset...")
    captions_df = pd.read_csv(config.CAPTIONS_FILE)
    captions_dict = captions_df.groupby('image')['caption'].apply(list).to_dict()

    print("Building vocabulary...")
    word2idx, idx2word, embedding_matrix = build_vocab(
        captions_dict,
        glove_embeddings,
        embedding_dim=config.EMBEDDING_DIM
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Creating dataset and dataloader...")
    dataset = FlickrDataset(
        captions_dict,
        config.IMAGE_DIR,
        transform,
        word2idx,
        config.MAX_LEN
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=None
    )

    print("Initializing the model...")
    model = ImageCaptioningModel(
        embed_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        vocab_size=len(word2idx),
        pretrained_embeddings=embedding_matrix
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting training...")
    train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.NUM_EPOCHS,
        checkpoint_dir=config.CHECKPOINT_DIR,
        device=config.DEVICE
    )

    # Test on example images in test directory
    print("\nTesting on example images...")
    test_images = [f for f in os.listdir(config.TEST_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for image_name in test_images:
        test_image_path = os.path.join(config.TEST_IMAGES_DIR, image_name)
        print(f"\nTesting image: {image_name}")
        test_example(
            test_image_path,
            model,
            transform,
            word2idx,
            idx2word,
            config.MAX_LEN
        )


if __name__ == "__main__":
    main()

