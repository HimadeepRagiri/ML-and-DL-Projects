import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(data_loader, num_images=16):
    """
    Visualize a batch of images from the DataLoader.
    """
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images[:num_images]

    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis("off")
    plt.show()
    print("Data visualization complete.")


def save_checkpoint(model, optimizer, epoch, checkpoint_path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    return start_epoch


def visualize_generated_images(generator, latent_dim, device, epoch, num_images=16):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        fake_images = generator(noise).cpu()
        fake_images = (fake_images.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]

        grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.axis("off")
        plt.title(f"Generated Images - Epoch {epoch}")
        plt.show()


def generate_and_plot_image(generator, latent_dim, device):
    # Set the generator to evaluation mode
    generator.eval()

    # Generate a random latent vector
    latent_vector = torch.randn(1, latent_dim, 1, 1, device=device)  # Shape: [1, latent_dim, 1, 1]

    # Generate a fake image
    with torch.no_grad():  # No gradients needed for inference
        fake_image = generator(latent_vector).cpu().squeeze(0)  # Shape: [C, H, W]

    # Post-process the image (scale from [-1, 1] to [0, 1])
    fake_image = (fake_image + 1) / 2  # Rescale to [0, 1]

    # Plot the image
    plt.figure(figsize=(4, 4))
    if fake_image.shape[0] == 1:  # Grayscale image
        plt.imshow(fake_image.squeeze(0), cmap="gray")
    else:  # RGB image
        plt.imshow(fake_image.permute(1, 2, 0))  # Shape: [H, W, C]
    plt.axis("off")
    plt.title("Generated Image")
    plt.show()
