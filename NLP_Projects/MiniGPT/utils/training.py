import torch
from tqdm import tqdm


def train_model(model, train_loader, optimizer, criterion, device, epochs):
    """
    Training loop with loss tracking and progress bars.
    """
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            # Prepare input and target
            # Target is input shifted by 1 position
            x = batch[:, :-1].to(device)
            target = batch[:, 1:].to(device)

            # Forward pass
            output = model(x)
            loss = criterion(output.reshape(-1, output.shape[-1]),
                             target.reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses