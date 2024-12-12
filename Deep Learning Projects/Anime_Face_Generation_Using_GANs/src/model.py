import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, feature_maps):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(feature_maps * 8, affine=True),
            nn.ReLU(True),

            # State: (feature_maps * 8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 4, affine=True),
            nn.ReLU(True),

            # State: (feature_maps * 4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.ReLU(True),

            # State: (feature_maps * 2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps, affine=True),
            nn.ReLU(True),

            # State: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: image_channels x 64 x 64
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, image_channels, feature_maps):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: image_channels x 64 x 64
            nn.Conv2d(image_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State: feature_maps x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (feature_maps * 2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (feature_maps * 4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (feature_maps * 8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        # Convolutional layers
        x = self.model(x)

        # Global average pooling to reduce to a single scalar per image
        return x.view(x.size(0), -1).mean(dim=1)
