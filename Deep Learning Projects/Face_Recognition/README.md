# Face Recognition System

## Project Overview

This is a comprehensive face recognition system that provides multiple functionalities including face enrollment, recognition, and real-time webcam-based identification. The project is designed to be flexible, easy to use, and can be deployed both locally and via web interface.

## Features

- 📸 Face Enrollment
- 🔍 Face Recognition
- 🖥️ Webcam Real-time Recognition
- 🌐 Web Deployment with Flask
- 📊 JSON-based Face Database Storage

## Project Structure

```
Face_Recognition/
│
├── README.md
├── requirements.txt
├── face_recognition/
│   ├── enrollment.py      # Face enrollment logic
│   ├── recognition.py     # Face recognition algorithms
│   └── utils.py           # Utility functions for database management
│
├── scripts/
│   ├── run_enrollment.py  # Script to enroll faces from command line
│   ├── run_recognition.py # Script to recognize faces from command line
│   └── webcam_recognition.py # Real-time webcam recognition script
│   
└── deployment/
    ├── app.py  # Web deployment Using Flask
    └── requirements.txt
```

## Prerequisites

- Python 3.8+
- OpenCV
- face-recognition library
- Flask (for web deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
cd Face_Recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Face Enrollment

Enroll a face from the command line:
```bash
python scripts/run_enrollment.py path/to/image.jpg "Person Name"
```

### Face Recognition

Recognize a face from an image:
```bash
python scripts/run_recognition.py path/to/image.jpg
```

### Webcam Recognition

Start real-time face recognition:
```bash
python scripts/webcam_recognition.py
```

### Web Deployment

Run app.py in the `deployment/` directory or use:
```bash
flask run
```

## Web Interface Features

- Enroll new faces using webcam
- Real-time face recognition
- Reset enrolled data
- Public URL generation with ngrok

## Technical Details

- Uses `face_recognition` library for face encoding
- Stores face encodings in JSON database
- Supports multiple face enrollment and recognition
- Utilizes OpenCV for image processing

## Performance Considerations

- Accuracy depends on image quality and lighting
- Recommended to use clear, front-facing images for enrollment

## Security Notes

- Face data is stored locally in JSON
- No cloud storage or external transmission of face data
- Recommended to implement additional authentication for production use

## Future Improvements

- [ ] Add face recognition confidence threshold
- [ ] Implement user authentication
- [ ] Support multiple face detection
- [ ] Create more robust error handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

