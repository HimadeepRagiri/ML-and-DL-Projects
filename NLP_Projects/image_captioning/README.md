# Image Captioning with Deep Learning

An implementation of image captioning using CNN-LSTM architecture with pretrained ResNet18 and GloVe embeddings. This model can generate natural language descriptions for images.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Project Overview
This project implements an image captioning system that combines a CNN (ResNet18) for image feature extraction with an LSTM for caption generation. The model is trained on the Flickr8k dataset and uses pretrained GloVe embeddings for word representations.

## Features
- Pretrained ResNet18 for image feature extraction
- LSTM-based caption generation
- GloVe word embeddings integration
- Checkpoint saving and loading
- Batch training with progress bars
- Customizable hyperparameters
- Easy-to-use inference pipeline

## Project Structure
```
image_captioning/
├── data/
│   ├── images/          # Place your Flickr8k images here
│   ├── captions.txt     # Place your captions file here
│   ├── glove.6B.300d.txt # Place GloVe embeddings here
│   └── dataset_info.md  # Dataset download instructions
├── checkpoints/         # Model checkpoints will be saved here
├── config.py           # Configuration parameters
├── data_preprocessing.py # Data loading and preprocessing
├── model.py            # Neural network architecture
├── utils.py            # Utility functions
├── train.py            # Training loop
├── main.py            # Main execution file
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Requirements
- Python 3.7+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- See requirements.txt for complete list

## Installation
1. Clone the repository:
```bash
git clone --no-checkout https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
cd ML-and-DL-Projects
git sparse-checkout init --cone
git sparse-checkout set NLP_Projects/image_captioning
cd NLP_Projects/image_captioning
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset Setup
1. Follow instructions in `data/dataset_info.md` to download:
   - Flickr8k dataset
   - GloVe embeddings
2. Verify the data directory structure matches the project structure shown above

## Usage
1. Configure parameters in `config.py` if needed
2. Run training:
```bash
python main.py
```

3. For inference on new images:
```python
from utils import test_example
test_example("path/to/your/image.jpg", model, transform, word2idx, idx2word, max_len)
```

## Model Architecture
- **Image Encoder**: ResNet18 (pretrained)
- **Word Embeddings**: GloVe 300d
- **Caption Generator**: LSTM
- **Training Strategy**: Teacher forcing
- **Loss Function**: Cross-entropy

## Training
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Epochs**: 10 (configurable)
- **Device**: Automatically uses GPU if available

Monitor training progress:
- Loss values are displayed in real-time
- Checkpoints are saved after each epoch
- Training can be resumed from checkpoints

## Results
The model typically achieves:
- Coherent caption generation
- Proper subject-verb agreement
- Relevant object detection and description
- Natural language structure

**Important Note:**
This project is primarily designed to demonstrate the functionality and implementation of an image captioning system. Due to computational limitations, extensive training results are not provided in this repository. However, the code is fully functional and users can:
- Train the model on their own hardware
- Monitor the training progress through decreasing loss values
- Observe the model's improvement in caption quality over epochs
- Experiment with different hyperparameters to achieve better results

For optimal training results, it is recommended to:
- Use a CUDA-capable GPU
- Train for more epochs (>10)
- Experiment with batch sizes based on available memory
- Monitor validation loss to prevent overfitting

---
For questions or issues, please open an issue in the GitHub repository.
