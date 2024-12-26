# Deep Learning Movie Recommendation System

## Overview
This project implements a Neural Matrix Factorization-based movie recommendation system using PyTorch. The model is trained on the MovieLens 1M dataset to predict user ratings and provide personalized movie recommendations.

## Project Structure
```
DeepLearning_Recommendation_System/
├── dataset/
│   └── ml-1m/
│       ├── movies.dat
│       └── ratings.dat
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── utils.py
├── notebook/
│   └── DeepLearning_Recommendation_System.ipynb
├── checkpoint_epoch_10.pt
├── main.py
├── requirements.txt
└── README.md
```

## Dataset
The project uses the MovieLens 1M dataset which includes:
- 1 million ratings from 6000 users on 4000 movies
- Each user has rated at least 20 movies
- Simple demographic info for users (age, gender, occupation, zip)

### Data Files
- `movies.dat`: Contains movie information (MovieID::Title::Genres)
- `ratings.dat`: Contains user ratings (UserID::MovieID::Rating::Timestamp)

## Features
- Neural Matrix Factorization (NeuMF) implementation
- User and movie embeddings
- Customizable embedding dimensions
- Training with validation and test splits
- Checkpoint saving and loading
- Movie recommendation generation
- Rating distribution visualization
- Jupyter notebook implementation

## Requirements
- Python 3.7+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- Pandas 1.3.0+
- scikit-learn 0.24.2+
- tqdm 4.62.0+
- matplotlib 3.4.2+

## Installation

1. Clone the repository:

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Google Colab Notebook
The project includes a Jupyter notebook (`notebook/DeepLearning_Recommendation_System.ipynb`) that demonstrates the implementation in Google Colab. This notebook provides an interactive way to:
- Understand the data preprocessing steps
- Visualize the data
- Train the model
- Generate recommendations
- Experiment with different parameters

### Training the Model

Run the main script to train the model:
```bash
python main.py
```

The script will:
1. Load and preprocess the MovieLens dataset
2. Initialize the neural network model
3. Train the model for 10 epochs
4. Save checkpoints during training
5. Evaluate the model on validation and test sets
6. Generate sample recommendations

### Code Structure

#### `data_preprocessing.py`
- `load_data()`: Loads and reads the MovieLens dataset
- `preprocess_data()`: Handles data preprocessing and train/val/test splitting
- `MovieDataset`: Custom PyTorch Dataset class for movie ratings
- `create_dataloaders()`: Creates DataLoader objects for training

#### `model.py`
- `RecommenderNet`: Neural Matrix Factorization model implementation
- Implements user and movie embeddings
- Forward pass logic for rating prediction

#### `utils.py`
- Model training utilities
- Checkpoint management
- Evaluation functions
- Visualization tools
- Recommendation generation

## Model Architecture

The Neural Matrix Factorization model consists of:
- User embedding layer
- Movie embedding layer
- Element-wise product of embeddings
- Linear layer for final prediction

### Hyperparameters
- Embedding dimension: 50
- Learning rate: 0.001
- Batch size: 512
- Training epochs: 10
- Optimizer: Adam
- Loss function: MSE

## Training Process

The model training includes:
1. Batch processing of user-movie interactions
2. Forward pass for rating prediction
3. Loss calculation and backpropagation
4. Regular checkpoint saving
5. Validation after each epoch
6. Final testing on held-out test set

## Results

The model is evaluated using Mean Squared Error (MSE) loss:
- Training loss is monitored during training
- Validation loss helps prevent overfitting
- Test loss provides final model performance metric

## Generating Recommendations

The system can generate movie recommendations for any user:
```python
recommended_movies = recommend_movies(
    model=model,
    user_id=user_id,
    movie_encoder=movie_encoder,
    device=device,
    top_n=10,
    movies_df=movies
)
```

## Saved Model

The trained model checkpoint (`checkpoint_epoch_10.pt`) contains:
- Model state dict
- Optimizer state dict
- Training epoch information

## Future Improvements

Potential enhancements for the project:
- Implementation of additional recommendation algorithms
- Integration of movie content features
- Addition of cross-validation
- Hyperparameter tuning
- Web interface for recommendations
- Support for online learning
- Integration of attention mechanisms
- Incorporation of temporal features

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- MovieLens dataset provided by GroupLens Research
- Neural Matrix Factorization implementation inspired by research papers
- PyTorch community for excellent documentation and tutorials

