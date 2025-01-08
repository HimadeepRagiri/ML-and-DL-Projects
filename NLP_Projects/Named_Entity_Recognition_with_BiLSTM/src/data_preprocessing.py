import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import torchtext
from torchtext.data.utils import get_tokenizer
import spacy

class CustomNERDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx, 0]
        tags = self.data.iloc[idx, 1].split()
        tokens = self.tokenizer(sentence)

        if len(tokens) > len(tags):
            tokens = tokens[:len(tags)]
        elif len(tags) > len(tokens):
            tags = tags[:len(tokens)]

        return tokens, tags

def parse_conll_file(filename):
    sentences = []
    tags = []
    with open(filename, "r") as f:
        sentence = []
        tag_seq = []
        for line in f:
            if line == "\n":
                sentences.append(" ".join(sentence))
                tags.append(" ".join(tag_seq))
                sentence = []
                tag_seq = []
            else:
                splits = line.strip().split()
                sentence.append(splits[0])
                tag_seq.append(splits[-1])
    return sentences, tags

def save_to_csv(sentences, tags, filename):
    df = pd.DataFrame({"sentence": sentences, "tags": tags})
    df.to_csv(filename, index=False)

def build_vocab(dataset):
    token_counter = Counter()
    tag_counter = Counter()

    for tokens, tags in dataset:
        token_counter.update(tokens)
        tag_counter.update(tags)

    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    TEXT = torchtext.vocab.vocab(token_counter, min_freq=1, specials=special_tokens)
    TAGS = torchtext.vocab.vocab(tag_counter, min_freq=1, specials=['<pad>'])

    TEXT.set_default_index(TEXT.get_stoi()['<unk>'])
    TAGS.set_default_index(TAGS.get_stoi()['<pad>'])

    return TEXT, TAGS

def create_data_loaders(train_dataset, val_dataset, test_dataset, TEXT, TAGS, batch_size):
    def collate_fn(batch):
        sentences, tags = zip(*batch)
        lengths = [len(s) for s in sentences]
        max_len = max(lengths)

        padded_sentences = [
            [TEXT[word] for word in sentence] +
            [TEXT['<pad>']] * (max_len - len(sentence))
            for sentence in sentences
        ]

        padded_tags = [
            [TAGS[tag] for tag in tag_seq] +
            [TAGS['<pad>']] * (max_len - len(tag_seq))
            for tag_seq in tags
        ]

        return (
            torch.tensor(padded_sentences, dtype=torch.long),
            torch.tensor(padded_tags, dtype=torch.long)
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader