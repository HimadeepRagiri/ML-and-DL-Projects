# Install necessary dependencies
import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64

# Flask Setup
app = Flask(__name__)

# Generator Model (same as in your code)
class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, feature_maps):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(feature_maps * 8, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 4, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load pre-trained model
latent_dim = 100
image_channels = 3
feature_maps = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "dcgan_checkpoint.pth"

# Initialize the generator
generator = Generator(latent_dim, image_channels, feature_maps)
generator.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
generator.eval()
generator.to(device)

# Function to generate an image
def generate_anime_image():
    latent_vector = torch.randn(1, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_image = generator(latent_vector).cpu().squeeze(0)
    fake_image = (fake_image + 1) / 2  # Normalize to [0, 1]

    # Convert the generated image to base64 for embedding in HTML
    buffer = BytesIO()
    fake_image = fake_image.permute(1, 2, 0).numpy()  # Convert to [H, W, C]
    im = Image.fromarray((fake_image * 255).astype(np.uint8))
    im.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str

# HTML Template for Flask WebApp
html_code = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anime Face Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
        }
        h1 {
            color: #4CAF50;
        }
        .image-container {
            margin: 20px;
        }
        .image-container img {
            width: 300px;
            height: 300px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Anime Face Generator</h1>
    <p>Click the button below to generate a random anime face.</p>
    <div class="image-container" id="generated-image-container">
        <img id="generated-image" src="" alt="Generated Anime Face" />
    </div>
    <button onclick="generateImage()">Generate Anime Face</button>

    <script>
        function generateImage() {
            fetch('/generate')
                .then(response => response.json())
                .then(data => {
                    const imgElement = document.getElementById('generated-image');
                    imgElement.src = 'data:image/png;base64,' + data.image;
                })
                .catch(error => {
                    console.error('Error generating image:', error);
                });
        }
    </script>
</body>
</html>
'''

# Create an HTML file on the fly for Flask (without needing to create directories)
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w") as f:
    f.write(html_code)

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("ngrok tunnel URL:", public_url)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    img_str = generate_anime_image()
    return jsonify({'image': img_str})

# Run Flask app
app.run()
