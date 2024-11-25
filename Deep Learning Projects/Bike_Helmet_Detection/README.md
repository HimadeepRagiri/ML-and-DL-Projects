# Bike Helmet Detection Project 🏍️

An AI-powered computer vision system that detects whether motorcycle riders are wearing helmets, built using YOLOv11 and deployed as a web application.

## 📋 Project Overview

This project implements a real-time helmet detection system for motorcycle riders using the YOLO (You Only Look Once) object detection framework. The system can process both images and videos, making it useful for traffic monitoring and safety compliance verification.

## 🚀 Features

- Real-time helmet detection in images and videos
- Web-based interface for easy interaction
- Support for multiple file formats (images: jpg, png; videos: mp4, avi, mov, mkv)
- High-accuracy detection using YOLOv11 architecture
- Interactive visualization of detection results

## 🛠️ Technical Stack

- **Deep Learning Framework:** YOLOv11 (Ultralytics)
- **Dataset Management:** Roboflow
- **Backend:** Flask
- **Deployment:** ngrok for tunnel access
- **Development Environment:** Google Colab / Local Python environment

## 📦 Installation

1. Clone the repository:
```bash
git clone [https://github.com/HimadeepRagiri/ML-and-DL-Projects.git]
cd Bike_Helmet_Detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
Bike_Helmet_Detection/
│
├── app.py                      # Flask deployment application
├── best.pt                     # Trained YOLO model weights
├── Bike_Helmet_Detection.ipynb # Training notebook
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🔧 Model Training

The model was trained using the following configuration:
- Base Model: YOLOv11
- Dataset: Custom motorcycle helmet dataset from Roboflow
- Training Parameters:
  - Epochs: 40
  - Image Size: 640x640
  - Confidence Threshold: 0.25

## 🌐 Web Application Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface through the provided ngrok URL

3. Upload an image or video file

4. View the detection results directly in your browser

## 📊 Model Performance

The model has been trained and validated on a custom dataset, showing strong performance in detecting:
- Motorcycle riders
- Properly worn helmets
- Missing helmets

Performance metrics and visualizations are available in the training notebook, including:
- Precision-Recall curves
- F1 score curves
- Confusion matrix
- Training batch visualizations

## 💻 API Reference

### Endpoints

#### `/` (GET)
- Home page with file upload interface

#### `/upload` (POST)
- Accepts multipart form data with a file
- Supports image and video files
- Returns HTML page with detection results

## 🔑 Environment Setup

Required environment variables:
```
ROBOFLOW_API_KEY=[your-api-key]
```

## 📝 Dependencies

Main dependencies include:
- ultralytics
- flask
- pyngrok
- roboflow
- torch
- opencv-python

For a complete list, see `requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Ultralytics for the YOLO implementation
- Roboflow for dataset management tools
- The open-source community for various tools and libraries
