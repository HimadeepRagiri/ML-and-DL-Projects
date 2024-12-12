# Dog Breed Classifier 🐾  
A deep learning-based project to classify images of dogs into their respective breeds. This project uses the **Stanford Dogs Dataset**, implements the **ResNet-50** architecture in PyTorch, and provides a user-friendly web application for deployment using Flask.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset Details](#dataset-details)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Model Training](#model-training)  
5. [Results and Evaluation](#results-and-evaluation)  
6. [Deployment](#deployment)  
7. [Dependencies](#dependencies)  
8. [Project Structure](#project-structure)  
9. [Steps to Run](#steps-to-run)  

---

## Project Overview  

The **Dog Breed Classifier** project identifies the breed of a dog from an image.  
It uses the **ResNet-50** architecture, a pre-trained model from PyTorch, and modifies the final output layer to classify images into 120 dog breeds from the **Stanford Dogs Dataset**. The project includes:  
- A Google Colab notebook for training and evaluation.  
- Flask-based web deployment for real-world usage.  
- Pre-trained model for quick inference.  

---

## Dataset Details  

- **Name**: Stanford Dogs Dataset  
- **Link**: [Stanford Dogs on Kaggle](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)  
- **Classes**: 120 dog breeds  
- **Images**: Contains thousands of labeled images for training and testing.  

**Steps to Download the Dataset**:  
1. Visit the [Kaggle link](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset).  
2. Download and extract the dataset.  
3. Place the images in the `data/` directory.  

Refer to `data/dataset_info.md` for detailed instructions.

---

## Data Preprocessing  

Data preprocessing includes:  
- Resizing images to `224x224`.  
- Normalization using ImageNet mean and standard deviation.  
- Data augmentation for better generalization.  

Scripts for preprocessing are available in `src/data_preprocessing.py`.  

---

## Model Training  

- **Framework**: PyTorch  
- **Model**: **ResNet-50** (pre-trained) with the final output layer modified to classify into 120 dog breeds.  
- **Loss Function**: Cross-Entropy Loss  
- **Optimizer**: Adam with learning rate scheduling using `ReduceLROnPlateau`.  
- **Number of Classes**: 120 (one for each breed).  

Model training is implemented in `notebooks/DogBreedClassifier.ipynb`. The notebook includes all steps:  
1. Data loading and augmentation  
2. Model architecture  
3. Training loop and evaluation  

---

## Results and Evaluation  

The model evaluation was done using the following metrics:
- **Test Loss**
- **Accuracy**
- **F1-Score**
- **Classification Report**  

These evaluations are computed using the `test_model()` function, which returns these metrics after running the model on the test dataset.

---

## Deployment  

The project provides a web application for real-world usage.  
- **Framework**: Flask  
- **Public Access**: Ngrok is used to provide a public URL for local deployment.  

Steps to deploy:  
1. Install dependencies in `deployment/requirements.txt`.  
2. Start the Flask app (`app.py`) in the `deployment/` directory.  
3. Ngrok generates a public URL for accessing the app.  

The application allows users to upload an image and predicts the dog's breed with its name and the uploaded image displayed.  

---

## Dependencies    

This project requires the following libraries:

- `torch`
- `torchvision`
- `scipy`
- `Pillow`
- `tqdm`
- `scikit-learn`

Install all required packages with: `pip install -r requirements.txt`

## Project Structure

- `data/`: This folder contains a **`dataset_info.md`** file with instructions on how to download the Stanford Dogs dataset from Kaggle. The raw data files are not included in the repository due to size constraints.
- `notebooks/`: Contains a Jupyter Notebook named **`DogBreedClassifier.ipynb`**, which includes the full implementation of the project, including data exploration, model training, and evaluation.
- `src/`: Contains Python scripts for data preprocessing, model architecture, and utility functions.
- `main.py`: Main script to execute the complete pipeline for training and evaluating the dog breed classification model.
- `model.pth`: Saved model checkpoint (trained model) for inference or further training.
- `deployment/`: Contains files for deploying the trained model as a web application using Flask, including the **`app.py`** file, **`model.pth`** for inference, and the **`requirements.txt`** file for deployment dependencies.

## Steps to Run

Follow these steps to run the project on your local machine:

1. **Clone the Repository**:
   - Clone the project repository to your local machine:
     ```bash
     git clone <repository_url>
     ```

2. **Install Dependencies**:
   - Install the required libraries by running the following command:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset**:
   - Ensure that you have the Stanford Dogs dataset in the `data/` directory. If not, download the dataset from the source (using the instructions in `data/dataset_info.md`) and place it in the `data/` folder.

4. **Run the Jupyter Notebook** (Optional):
   - If you prefer to explore the data and model in a Jupyter Notebook, navigate to the `notebooks/` directory and open the notebook file **`DogBreedClassifier.ipynb`**:
     ```bash
     jupyter notebook
     ```
   - The notebook includes:
     - **Data Preprocessing**: Loading and processing the dataset, including resizing and normalizing images.
     - **Model Training**: Defining the model architecture, training the neural network, and evaluating its performance.
     - **Model Deployment**: Steps to deploy the trained model as a web application.

5. **Running the Model Training Using Python Files**:

   The project consists of several Python files that handle different stages of the pipeline. Here's how to run them:

   1. **Data Preprocessing**:
      - The preprocessing steps are defined in the `src/data_preprocessing.py` file. You can import and call the functions from this file in `main.py` to preprocess the dataset.
      - It handles:
        - Loading and processing images.
        - Resizing images to fit the model.
        - Normalizing image data.

   2. **Utility Functions**:
      - The `src/utils.py` file contains utility functions such as `load_checkpoint()` for loading saved models and checkpoints, `test_model()` for evaluating the model predictions, as well as other helper functions for logging and saving models.

   3. **Model Building**:
      - The model architecture is defined in the `src/model.py` file. This file contains the neural network architecture built using PyTorch, defining the layers and the forward pass.
      - You can modify this file to experiment with different network architectures if needed.
   
   4. **Training and Evaluation**:
      - The main model training logic is handled in `main.py`. This script:
        - Loads and preprocesses the dataset.
        - Initializes the model, optimizer, and loss function.
        - Trains the model by calling functions from `src/model.py` and `src/data_preprocessing.py`.
        - Evaluates the model's performance using metrics like accuracy.
        - Saves the trained model checkpoint.

   5. **Loading a Saved Model**:
      - If you want to load a previously trained model and make predictions, you can use the `load_checkpoint()` function from `src/utils.py` in `main.py`. Example:
        ```python
        checkpoint_path = 'path_to_saved_model/model.pth'
        model, optimizer, scheduler, epoch, loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        ```

6. **Run the `main.py` Script**:
   - To train the neural network model, run the `main.py` script:
     ```bash
     python main.py
     ```
   - The script will:
     - Preprocess the data using the functions defined in `src/data_preprocessing.py`.
     - Train the neural network model using the architecture defined in `src/model.py`.
     - Evaluate the trained model and print the results like (accuracy, loss) using functions defined in `src/utils.py`.
     - Save the model checkpoint.

7. **Model Inference**:
   - Once the model is trained and the checkpoint is saved, you can use the trained model for inference. To load the model and make predictions on new images, use the following code in `app.py` (for deployment):
     ```python
     model = load_checkpoint(model, optimizer, scheduler, 'deployment/model.pth')
     ```

8. **Running the Web Deployment**:
   - To deploy the model and make predictions through a web interface, navigate to the `deployment/` directory and run the following command:
     ```bash
     python app.py
     ```
   - This will start a Flask web application, and the model will be accessible at a publicly accessible URL through ngrok.

9. **Experiment and Fine-Tune**:
   - You can experiment with different hyperparameters such as learning rate, batch size, and number of epochs to improve the model's performance.
   - Additionally, you can modify the model architecture in `src/model.py` to try different deep learning architectures.

**Note**: Ensure that your Python environment has the necessary dependencies installed, and that the dataset is properly set up before running the scripts.

