import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def load_and_preprocess_data(data_dir, image_size=128, batch_size=64):
    """
    Load and preprocess the dataset from a single folder of images.
    """
    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, 0  # Returning 0 as dummy label since we don't need labels.

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] for GANs
    ])

    # Create dataset
    dataset = CustomImageDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"Loaded {len(dataset)} images from {data_dir}")
    return data_loader
