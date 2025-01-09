import torch
import os
from PIL import Image

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss

def generate_caption(model, image, word2idx, idx2word, max_len, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        feature = model.encoder(image.unsqueeze(0))
        caption = [word2idx["< SOS >"]]
        for _ in range(max_len):
            input_seq = torch.tensor(caption, device=device).unsqueeze(0)
            embedding = model.embedding(input_seq)
            output, _ = model.lstm(torch.cat((feature.unsqueeze(1), embedding), dim=1))
            pred = output[:, -1, :].argmax(1).item()
            caption.append(pred)
            if pred == word2idx["<EOS>"]:
                break
        return " ".join(idx2word[idx] for idx in caption[1:])

def test_example(image_path, model, transform, word2idx, idx2word, max_len):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image)
    caption = generate_caption(model, transformed_image, word2idx, idx2word, max_len, next(model.parameters()).device)
    print(f"Generated Caption: {caption}")

