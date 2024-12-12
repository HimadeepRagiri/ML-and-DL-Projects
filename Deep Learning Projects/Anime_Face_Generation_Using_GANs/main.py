import torch
from data_preprocessing import load_and_preprocess_data
from model import Generator, Discriminator
from train import train_dcgan
from utils import visualize_data, generate_and_plot_image

# Hyperparameters
latent_dim = 100
image_channels = 3
feature_maps = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and Preprocess Data
data_dir = "<INSERT_PATH_HERE>"  # TODO: Replace this with the path to your dataset folder
data_loader = load_and_preprocess_data(data_dir)
visualize_data(data_loader)

# Initialize models
generator = Generator(latent_dim, image_channels, feature_maps)
discriminator = Discriminator(image_channels, feature_maps)

# Checkpoint
checkpoint_path = "dcgan_checkpoint.pth"

# Train DCGAN
train_dcgan(generator, discriminator, data_loader, num_epochs, device, latent_dim, lr, beta1, checkpoint_path)

# Generating an image
generate_and_plot_image(generator, latent_dim=100, device=device)
