import torch
import torch.nn as nn
from tqdm import tqdm

from data_preprocessing import load_and_preprocess_data
from model import Generator, Discriminator
from utils import save_checkpoint, load_checkpoint, visualize_generated_images

def train_dcgan(generator, discriminator, data_loader, num_epochs, device, latent_dim, lr=0.0002, beta1=0.5, checkpoint_path="dcgan_checkpoint.pth"):
    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Move models to the device
    generator.to(device)
    discriminator.to(device)

    # Resume from checkpoint if available
    start_epoch = 0
    try:
        start_epoch = load_checkpoint(checkpoint_path, generator, optimizer_g)
        print(f"Resuming training from epoch {start_epoch}.")
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for real_images, _ in progress_bar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Labels
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # =======================
            # Train Discriminator
            # =======================
            optimizer_d.zero_grad()

            # Real images
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # =======================
            # Train Generator
            # =======================
            optimizer_g.zero_grad()

            # Fake images through discriminator
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_g.step()

            # Accumulate loss for progress display
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

            progress_bar.set_postfix({"D Loss": d_loss.item(), "G Loss": g_loss.item()})

        # Visualize generated images at the end of the epoch
        visualize_generated_images(generator, latent_dim, device, epoch+1)

        # Save checkpoint
        save_checkpoint(generator, optimizer_g, epoch+1, checkpoint_path)

        print(f"Epoch {epoch+1} completed: Generator Loss = {g_loss_epoch:.4f}, Discriminator Loss = {d_loss_epoch:.4f}")
