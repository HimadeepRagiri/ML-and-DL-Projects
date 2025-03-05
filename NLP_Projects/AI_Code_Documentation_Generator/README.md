# AI Code Documentation Generator

## Overview
This project implements an AI-powered code documentation generator using GPT-2 with LoRA (Low-Rank Adaptation) fine-tuning. The system automatically generates documentation for Python code snippets, helping developers maintain better documented codebases.

![Deployment Interface](images/Deployment_Image.png)

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Contributing](#contributing)

## Features
- Automatic documentation generation for Python code snippets
- Fine-tuned GPT-2 model using LoRA for efficient training
- Web interface using Gradio for easy interaction
- Support for both CPU and GPU inference
- Configurable model parameters and generation settings
- Comprehensive training pipeline with TensorBoard logging

## Project Structure
```
AI_Code_Documentation_Generator/
├── config/
│   └── config.py               # Configuration parameters
├── src/
│   ├── data/
│   │   └── data_processing.py  # Data loading and preprocessing
│   ├── model/
│   │   ├── model.py           # Model architecture and inference
│   │   └── tokenizer.py       # Tokenizer initialization
│   ├── training/
│   │   └── trainer.py         # Training loop and evaluation
│   └── utils/
│       └── utils.py           # Utility functions
├── app/
│   ├── app.py                 # Gradio web interface
│   ├── DOWNLOAD_MODEL.md      # Model download instructions
│   └── trained_model/         # Saved model weights
├── notebooks/
│   └── AI_Code_Documentation_Generator.ipynb  # Training notebook
├── images/
│   └── Deployment_Image.png   # UI screenshot
├── main.py                    # Main training script
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
   git clone --no-checkout https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
   cd ML-and-DL-Projects
   git sparse-checkout init --cone
   git sparse-checkout set NLP_Projects/AI_Code_Documentation_Generator
   cd NLP_Projects/AI_Code_Documentation_Generator
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the trained model:
   - Follow the instructions in `app/DOWNLOAD_MODEL.md` to download and set up the trained model
   - The model files need to be placed in the `app/trained_model` directory

5. Verify the installation:
   ```bash
   python -c "from transformers import GPT2Tokenizer; tokenizer = GPT2Tokenizer.from_pretrained('app/trained_model')"
   
## Usage

### Web Interface
1. Start the Gradio interface:
```bash
python -m app.app
```
2. Open the provided URL in your browser
3. Paste your Python code snippet
4. Click "Submit" to generate documentation

### Programmatic Usage
```python
from src.model.model import generate_documentation
from transformers import GPT2Tokenizer
from peft import PeftModel

# Load model and tokenizer
model = PeftModel.from_pretrained("app/trained_model")
tokenizer = GPT2Tokenizer.from_pretrained("app/trained_model")

# Generate documentation
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
documentation = generate_documentation(code, model, tokenizer)
print(documentation)
```

## Training

The model was trained on the CodeSearchNet dataset using the following steps:

1. Data preparation:
   - Loading Python functions from CodeSearchNet
   - Extracting docstrings and code
   - Preprocessing and tokenization

2. Model configuration:
   - Base model: GPT-2
   - LoRA adaptation for efficient fine-tuning
   - Special tokens for code and documentation

3. Training parameters:
   - Epochs: 2
   - Learning rate: 5e-5
   - Batch size: 4
   - Gradient accumulation steps: 4

To train the model:
```bash
python main.py
```

## Model Performance

Current model metrics:
- Evaluation Loss: 1.8152
- Perplexity: 6.14

Note: Due to computational limitations, the model was trained for only 2 epochs. While it produces decent results, the generation quality could be significantly improved with:
- More training epochs
- Larger dataset
- Better hyperparameter tuning
- More computational resources

## Technical Details

### Model Architecture
- Base Model: GPT-2
- Adaptation: LoRA (Low-Rank Adaptation)
- Parameters:
  - LoRA rank: 8
  - LoRA alpha: 16
  - LoRA dropout: 0.1

### Data Processing
- Dataset: CodeSearchNet Python
- Training samples: 50,000
- Validation samples: 5,000
- Special tokens:
  - `<CODE>`: Marks the start of code
  - `<DOC>`: Marks the start of documentation
  - `<SEP>`: Separator token

### Training Infrastructure
- Framework: PyTorch
- Logging: TensorBoard, Weights & Biases
- Hardware: Google Colab

## Limitations and Future Improvements

Current Limitations:
1. Limited training data and epochs
2. Basic preprocessing pipeline
3. Generation quality needs improvement
4. Limited context window

Planned Improvements:
1. Train on larger dataset
2. Increase training epochs
3. Experiment with different model architectures
4. Implement better preprocessing
5. Add support for multiple programming languages
6. Improve documentation quality metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
⭐ If you find this project useful, please consider giving it a star on GitHub!
