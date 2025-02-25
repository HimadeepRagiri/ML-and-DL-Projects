# Named Entity Recognition with BiLSTM

## Overview
This project implements a Bidirectional LSTM (BiLSTM) model for Named Entity Recognition (NER) using PyTorch. The model is trained on the CoNLL-2003 dataset and can identify entities such as Person (PER), Organization (ORG), Location (LOC), and Miscellaneous (MISC) in text.

## Project Structure
```
Named_Entity_Recognition_with_BiLSTM/
├── data/
│   └── conll2003/
│       ├── eng.train    # Training data
│       ├── eng.testb    # Test set B
│       └── eng.testa    # Test set A
├── checkpoints/
│   └── model.pth        # Trained model weights
├── notebooks/
│   └── Named_Entity_Recognition_with_BiLSTM.ipynb  # Implementation notebook
├── src/
│   ├── data_preprocessing.py  # Data processing utilities
│   ├── model.py              # BiLSTM model architecture
│   ├── train.py             # Training procedures
│   ├── utils.py             # Utility functions
│   └── config.py            # Configuration parameters
├── main.py                  # Main training script
├── app.py                   # Flask web application
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- PyTorch 2.0.0
- TorchText 0.15.1
- SpaCy 3.5.3
- Flask 2.3.3
- pandas 2.0.3
- numpy 1.24.3
- pyngrok 6.0.0
- tqdm 4.65.0

## Installation

1. Clone the repository:
```bash
git clone --no-checkout https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
cd ML-and-DL-Projects
git sparse-checkout init --cone
git sparse-checkout set NLP_Projects/Named_Entity_Recognition_with_BiLSTM
cd NLP_Projects/Named_Entity_Recognition_with_BiLSTM
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the SpaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Data
The project uses the CoNLL-2003 dataset, which should be placed in the `data/conll2003/` directory. The dataset includes:
- `eng.train`: Training data
- `eng.testb`: Development set
- `eng.testa`: Test set

Each file follows the CoNLL format with space-separated values:
```
Word    POS    Chunk    NER-tag
```

## Model Architecture
The BiLSTM model consists of:
- Embedding layer
- Bidirectional LSTM layer
- Dropout layer (0.5)
- Fully connected layer

Key features:
- Bidirectional processing for context from both directions
- Word embeddings dimension: 100
- Hidden state dimension: 256
- Batch size: 64

## Training

1. To train a new model:
```bash
python main.py
```

The script will:
- Process the CoNLL-2003 dataset
- Train the BiLSTM model
- Save checkpoints to the `checkpoints/` directory
- Evaluate on validation and test sets
- Start the web server for predictions

Training parameters can be modified in `src/config.py`.

## Web Interface

The project includes a Flask web application for making predictions. To start the server:

1. Make sure you have a trained model in the `checkpoints/` directory
2. Run:
```bash
python main.py
```
3. Access the web interface through the provided ngrok URL

The interface features:
- Text input area for entering sentences
- Real-time NER predictions
- Color-coded entity highlighting
- Interactive visualization

## Module Description

### src/data_preprocessing.py
- `CustomNERDataset`: PyTorch dataset class for NER data
- `parse_conll_file`: Parses CoNLL format files
- `build_vocab`: Creates vocabularies for tokens and tags
- Data loading and preprocessing utilities

### src/model.py
- `BiLSTMNER`: Main model architecture
- Implements the bidirectional LSTM with word embeddings

### src/train.py
- Training loop implementation
- Evaluation functions
- Checkpoint management

### src/utils.py
- Utility functions for model operations
- Prediction functions
- Checkpoint saving/loading

### src/config.py
- Model hyperparameters
- Training configurations
- File paths and constants

## Notebook Usage

The `notebooks/Named_Entity_Recognition_with_BiLSTM.ipynb` contains:
- Detailed implementation walkthrough
- Model training process
- Evaluation metrics
- Example predictions
- Visualization of results

Open it in Google Colab or Jupyter Notebook for interactive exploration.

## Performance

The model achieves the following performance metrics after 5 epochs of training:
- Training Loss: 0.2310
- Validation Loss: 0.4273
- Test Loss: 0.3725

Note: Due to computational constraints, the model was trained for only 5 epochs. The performance metrics are decent for this limited training, but the model's accuracy could be significantly improved by training for more epochs (20-30 epochs recommended) with adequate computational resources. This implementation serves as a demonstration of the architecture and workflow; for production use, more extensive training is advised.

Example predictions:
```
Input: "John works at Microsoft in New York."
Output:
- John: B-PER
- works: O
- at: O
- Microsoft: B-ORG
- in: O
- New: B-LOC
- York: I-LOC
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- CoNLL-2003 dataset creators
- PyTorch team
