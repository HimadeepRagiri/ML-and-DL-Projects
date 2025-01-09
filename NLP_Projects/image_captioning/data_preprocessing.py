import spacy
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

spacy_nlp = spacy.load("en_core_web_sm")

def tokenize_with_spacy(caption):
    doc = spacy_nlp(caption.lower())
    tokens = [token.text for token in doc]
    return tokens

def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def build_vocab(captions_dict, glove_embeddings, min_freq=5, embedding_dim=300):
    all_tokens = []
    for captions in captions_dict.values():
        for caption in captions:
            tokens = tokenize_with_spacy(caption)
            all_tokens.extend(tokens)

    word_freq = Counter(all_tokens)
    vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "< SOS >")
    vocab.insert(2, "<EOS>")
    vocab.insert(3, "<UNK>")

    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim))
    for idx, word in enumerate(vocab):
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word, embedding_matrix

class FlickrDataset(Dataset):
    def __init__(self, captions_dict, image_dir, transform, word2idx, max_len):
        self.captions_dict = captions_dict
        self.image_dir = image_dir
        self.transform = transform
        self.word2idx = word2idx
        self.max_len = max_len
        self.image_ids = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        captions = self.captions_dict[img_id]

        img_id = img_id.replace('.jpg', '')
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        caption = captions[0]
        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokenize_with_spacy(caption)]
        tokens = [self.word2idx["< SOS >"]] + tokens[:self.max_len-2] + [self.word2idx["<EOS>"]]
        tokens += [self.word2idx["<PAD>"]] * (self.max_len - len(tokens))
        return image, torch.tensor(tokens)
