from flask import Flask, request
from pyngrok import ngrok
from ultralytics import YOLO
import os
import shutil
import glob

# Initialize Flask App
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"  # Folder to save uploaded files
app.config["DETECT_FOLDER"] = "static/detections"  # Folder to save detection results

# Create folders if they do not exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["DETECT_FOLDER"], exist_ok=True)

# Load YOLO Model
model = YOLO('best.pt')

@app.route("/")
def index():
    """Render the homepage."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YOLO Object Detection</title>
    </head>
    <body>
        <h1>YOLO Object Detection</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="file">Upload an Image or Video:</label>
            <input type="file" name="file" id="file" required>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """

@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload and run YOLO inference."""
    if "file" not in request.files:
        return "No file uploaded!", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected!", 400

    # Save the uploaded file
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(upload_path)

    # Check file type (image or video) by extension
    file_ext = file.filename.split(".")[-1].lower()
    is_video = file_ext in ["mp4", "avi", "mov", "mkv"]

    # Run YOLO inference
    results = model.predict(source=upload_path, save=True, save_txt=False, save_conf=True)

    # Get the latest prediction folder
    runs_path = os.path.join("runs", "detect")
    all_folders = sorted(glob.glob(os.path.join(runs_path, "predict*")), key=os.path.getmtime, reverse=True)

    if not all_folders:
        return "Prediction failed. No output found!", 500

    latest_folder = all_folders[0]  # Latest prediction folder

    if is_video:
        # Find the output video file
        predicted_files = glob.glob(os.path.join(latest_folder, "*.mp4")) + \
                          glob.glob(os.path.join(latest_folder, "*.avi"))
        if not predicted_files:
            return "Prediction failed. No output video found!", 500

        predicted_video_path = predicted_files[0]

        # Move the predicted video to the detection folder
        detection_path = os.path.join(app.config["DETECT_FOLDER"], file.filename)
        shutil.copy(predicted_video_path, detection_path)

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YOLO Detection Result</title>
        </head>
        <body>
            <h1>Detection Result</h1>
            <h2>Uploaded Video:</h2>
            <video controls style="max-width:100%;height:auto;">
                <source src="/static/uploads/{file.filename}" type="video/{file_ext}">
                Your browser does not support the video tag.
            </video>
            <h2>Detection Output:</h2>
            <video controls style="max-width:100%;height:auto;">
                <source src="/static/detections/{file.filename}" type="video/{file_ext}">
                Your browser does not support the video tag.
            </video>
            <br>
            <a href="/">Upload Another File</a>
        </body>
        </html>
        """
    else:
        # Find the output image file
        predicted_files = glob.glob(os.path.join(latest_folder, "*.jpg")) + \
                          glob.glob(os.path.join(latest_folder, "*.png"))
        if not predicted_files:
            return "Prediction failed. No output image found!", 500

        predicted_image_path = predicted_files[0]

        # Move the predicted image to the detection folder
        detection_path = os.path.join(app.config["DETECT_FOLDER"], file.filename)
        shutil.copy(predicted_image_path, detection_path)

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YOLO Detection Result</title>
        </head>
        <body>
            <h1>Detection Result</h1>
            <h2>Uploaded Image:</h2>
            <img src="/static/uploads/{file.filename}" alt="Uploaded Image" style="max-width:100%;height:auto;">
            <h2>Detection Output:</h2>
            <img src="/static/detections/{file.filename}" alt="Detection Result" style="max-width:100%;height:auto;">
            <br>
            <a href="/">Upload Another File</a>
        </body>
        </html>
        """

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("ngrok tunnel URL:", public_url)

if __name__ == "__main__":
    app.run()
