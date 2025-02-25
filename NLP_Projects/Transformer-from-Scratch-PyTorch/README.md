# Transformer from Scratch in PyTorch

This project provides a from-scratch implementation of the Transformer model, as described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), using **PyTorch**. The code is modular, allowing for easy understanding and testing of different components of the model.

## Table of Contents

1. [Introduction](#introduction)
2. [Directory Structure](#directory-structure)
3. [Transformer Architecture](#transformer-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Training (Optional)](#training-optional)
8. [Contributing](#contributing)

## Introduction

The Transformer model has revolutionized the field of deep learning and natural language processing (NLP) by introducing the concept of self-attention and completely eliminating the need for recurrent layers like RNNs and LSTMs. This innovative approach allows models to process input sequences in parallel, making them more efficient and scalable for large-scale tasks.

This project provides a from-scratch implementation of the Transformer model using **PyTorch**. The architecture behind the Transformer has become the foundation of modern **generative AI** systems like **ChatGPT** and other AI-driven technologies shaping the future of human-computer interaction.

By leveraging key mechanisms such as scaled dot-product attention, multi-head attention, and positional encoding, the Transformer model allows machines to understand and generate human-like text, images, and even music with remarkable accuracy. This implementation aims to provide a clear and modular structure for understanding how each part of the Transformer works—**attention mechanisms**, **feedforward networks**, and the **encoder-decoder architecture**.

The project’s modular design allows users to test and modify individual components, helping to provide a deeper understanding of each mechanism's role within the architecture. Whether you're looking to learn more about Transformer models or build upon this implementation for your own projects, this repository serves as an excellent starting point.

## Directory Structure

```
Transformer-from-Scratch-PyTorch/
│
├── attention.py       # Scaled Dot-Product Attention & Multi-Head Attention
├── feedforward.py     # Position-Wise Feedforward Network
├── positional.py      # Positional Encoding
├── encoder.py         # Encoder and EncoderLayer
├── decoder.py         # Decoder and DecoderLayer
├── transformer.py     # Main Transformer model
└── utils.py           # Utility functions (e.g., create_pad_mask, create_subsequent_mask)
│
├── requirements.txt   # Dependencies
├── README.md          # Project description
├── main.py            # Entry point for testing the model
└── Transformer-from-Scratch-PyTorch.ipynb   # Colab notebook containing notebook implementation
```

## Transformer Architecture

The Transformer model is based on the following core components:

1. **Scaled Dot-Product Attention**: A mechanism that computes the attention scores between the input tokens.
2. **Multi-Head Attention**: This allows the model to jointly attend to information from different representation subspaces at different positions.
3. **Position-Wise Feedforward Networks**: A simple 2-layer fully connected network applied to each position separately and identically.
4. **Positional Encoding**: Since the model does not have an inherent notion of sequence order, positional encodings are added to the input embeddings to retain the order information.
5. **Encoder-Decoder Structure**: The encoder reads the input sequence and outputs a context, while the decoder takes this context along with the target sequence and generates the output sequence.

### Key Modules

- **attention.py**: Contains the `ScaledDotProductAttention` and `MultiHeadAttention` classes that implement attention mechanisms.
- **feedforward.py**: Implements the `PositionwiseFeedforward` class, which represents the fully connected layers after attention mechanisms.
- **positional.py**: Contains the `PositionalEncoding` class, which is responsible for adding positional encodings to the input sequences.
- **encoder.py**: Defines the `Encoder` class and its components like `EncoderLayer`, which processes the input sequence with self-attention and feedforward layers.
- **decoder.py**: Contains the `Decoder` and `DecoderLayer` classes that process the target sequence with self-attention, cross-attention with the encoder's output, and feedforward layers.
- **transformer.py**: Implements the main `Transformer` class that combines the encoder and decoder to perform sequence-to-sequence tasks.
- **utils.py**: Includes helper functions like `create_pad_mask` and `create_subsequent_mask`, which are used to handle padding and masking in sequences.

## Installation

To use this project, you will need Python and PyTorch installed. You can install the required dependencies using the following steps:

### Prerequisites

- Python 3.6+
- PyTorch 1.9.0+

### Step 1: Clone the repository
```bash
git clone --no-checkout https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
cd ML-and-DL-Projects
git sparse-checkout init --cone
git sparse-checkout set NLP_Projects/Transformer-from-Scratch-PyTorch
cd NLP_Projects/Transformer-from-Scratch-PyTorch
```

### Step 2: Install dependencies

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

### Step 3: Run the model

You can now run the model by using the provided `main.py` to test the implementation.

## Usage

Once the installation is complete, you can use the Transformer model in two ways: **testing** or **training**.

### Running the Model in Colab

The `Transformer-from-Scratch-PyTorch.ipynb` notebook is a Jupyter Notebook implementation of the Transformer model. You can open this notebook in Google Colab and run it interactively.

- [Open the notebook in Google Colab](https://colab.research.google.com/github/yourusername/Transformer-from-Scratch-PyTorch/blob/main/Transformer-from-Scratch-PyTorch.ipynb)

## Testing

To test the implementation, you can run the `main.py` file, which runs a set of assertions on the model's output. This file performs a forward pass through the Transformer model and checks the output shape.

Run the tests with the following command:

```bash
python main.py
```

If the tests pass, the model is functioning correctly.

## Training (Optional)

If you'd like to train the model on a specific task (e.g., translation or text generation), you can modify the `main.py` file or create a custom training script.

1. Prepare your dataset (e.g., for machine translation or text summarization).
2. Modify the `main.py` script to load the dataset, train the model, and save the trained weights.
3. Use the `torch.optim` package to define an optimizer and the `torch.nn.CrossEntropyLoss` for the loss function.

## Contributing

We welcome contributions to this project. To contribute:

1. Fork this repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request with a detailed description of your changes.


