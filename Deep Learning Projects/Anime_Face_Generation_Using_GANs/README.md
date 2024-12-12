# Anime Face Generation Using Generative Adversarial Networks (GANs)

## Project Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic anime face images using PyTorch. The goal is to create a machine learning model capable of generating realistic anime character portraits from random noise vectors.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Deployment](#deployment)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Generate synthetic anime face images using Deep Convolutional GANs
- Flexible training pipeline
- Web application for interactive image generation
- Checkpointing and model saving
- Visualization utilities

## Project Structure

```
Anime_Face_Generation_Using_GANs/
│
├── data/
│   └── data_info.md
│
├── deployment/
│   ├── app.py
│   └── requirements.txt
│
├── notebooks/
│   └── project_implementation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

### Data Source
- Dataset: Anime Face Dataset from Kaggle
- Recommended: [Anime Face Dataset on Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset)

### Data Preparation
1. Download the dataset from Kaggle
2. Place images in the `data/` directory
3. Refer to `data/data_info.md` for detailed dataset instructions

## Training

### Configuration
Adjust hyperparameters in `main.py`:
- `latent_dim`: Dimension of random noise vector
- `image_channels`: Number of image color channels
- `feature_maps`: Base feature map size
- `num_epochs`: Training duration
- `lr`: Learning rate

### Running Training
```bash
python main.py
```

### Training Features
- Progressive image generation visualization
- Epoch-wise loss tracking
- Model checkpoint saving

## Deployment

### Web Application
A Flask-based web application is provided for interactive image generation.

#### Running the Web App
```bash
cd deployment
pip install -r requirements.txt
python app.py
```

## Model Architecture

### Generator
- Input: Random noise vector
- Layers: Transposed Convolutional Layers
- Normalization: Instance Normalization
- Activation: ReLU, Tanh

### Discriminator
- Input: Generated or Real Images
- Layers: Convolutional Layers
- Normalization: Batch Normalization
- Activation: LeakyReLU

## Results

### Generated Images
- Realistic anime face generation
- Variety in facial features
- Resolution: 64x64 pixels

### Training Metrics
- Image quality improvement over epochs

## Troubleshooting

- Ensure CUDA is properly installed for GPU acceleration
- Check dataset path in `main.py`
- Verify all dependencies are installed
- Low-quality images might indicate insufficient training

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Improvements

- Support higher resolution image generation
- Implement advanced GAN techniques
- Add more interactive web features
- Create a comprehensive evaluation metric

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Kaggle Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
- Inspiration from various GAN research papers

---

**Note**: Always ensure ethical use of generated images and respect copyright and intellectual property rights.