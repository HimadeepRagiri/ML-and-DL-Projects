from tqdm import tqdm
import os
from utils import save_checkpoint

def train_model(model, dataloader, criterion, optimizer, num_epochs, checkpoint_dir, device):
    model.train()
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, outputs.size(2)), captions.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

