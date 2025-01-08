import torch
import os
from tqdm import tqdm

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename="checkpoint.pth"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_dir, filename):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def predict(model, sentence, tokenizer, TEXT, TAGS, device):
    model.eval()
    tokens = tokenizer(sentence)
    token_indices = [TEXT[token] for token in tokens]
    token_tensor = torch.tensor([token_indices]).to(device)

    with torch.no_grad():
        predictions = model(token_tensor)
        predicted_indices = predictions.argmax(dim=2)[0]
        predicted_tags = [TAGS.get_itos()[idx.item()] for idx in predicted_indices]

    print("\nTokens and their predicted tags:")
    for token, tag in zip(tokens, predicted_tags):
        print(f"{token}: {tag}")

    return list(zip(tokens, predicted_tags))