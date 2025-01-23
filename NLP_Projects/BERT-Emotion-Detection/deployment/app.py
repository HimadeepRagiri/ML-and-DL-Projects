# Import libraries
from flask import Flask, render_template_string, request, jsonify
from pyngrok import ngrok
import torch
from transformers import BertTokenizer
from accelerate import Accelerator
import shutil
import os

# Initialize Flask app
app = Flask(__name__)

# Create directories
os.makedirs('static', exist_ok=True)

# Load model and tokenizer
accelerator = Accelerator()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    os.path.join(Config.CHECKPOINT_DIR, "best_model"),
    num_labels=Config.NUM_LABELS
)
model = accelerator.prepare(model)
model.eval()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            background: #f8f9fa;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .main-card {
            width: 100%;
            max-width: 800px;
            border: none;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: all 0.3s;
        }
        .header-section {
            background: #2563eb;
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .content-section {
            background: white;
            padding: 2rem;
            position: relative;
        }
        textarea {
            width: 100%;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1rem;
            resize: none;
            transition: all 0.3s;
        }
        #resultCard {
            display: none;
            margin-top: 2rem;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .emotion-display {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2563eb;
            margin: 1rem 0;
            padding: 1.5rem;
            background: #f0f4ff;
            border-radius: 15px;
            display: inline-block;
        }
        .confidence-badge {
            font-size: 1rem;
            color: #4b5563;
            margin-top: 0.5rem;
        }
        .btn-primary {
            background: #2563eb;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            transition: all 0.3s;
            width: 100%;
        }
        .btn-primary:hover {
            background: #1d4ed8;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="main-card">
        <div class="header-section">
            <h1>Emotion Detection</h1>
            <p>Transforming Text into Emotional Insights</p>
        </div>

        <div class="content-section">
            <form id="analysisForm" onsubmit="analyzeText(event)">
                <textarea name="text" id="inputText" 
                    rows="4" 
                    placeholder="Enter your text here..." 
                    required></textarea>
                <button type="submit" class="btn btn-primary mt-3">
                    Analyze Emotion
                </button>
            </form>

            <div id="resultCard" class="result-card">
                <div class="text-center">
                    <div class="emotion-display">
                        <span id="emotionResult"></span>
                        <div class="confidence-badge">
                            Confidence: <span id="confidenceResult"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function analyzeText(event) {
            event.preventDefault();
            const formData = new FormData(event.target);
            const response = await fetch('/predict', {
                method: 'POST',
                body: new URLSearchParams(formData)
            });

            const data = await response.json();

            if(data.error) {
                alert(data.error);
                return;
            }

            // Update results
            document.getElementById('emotionResult').textContent = data.emotion;
            document.getElementById('confidenceResult').textContent = data.confidence;

            // Show result card
            const resultCard = document.getElementById('resultCard');
            resultCard.style.display = 'block';
            setTimeout(() => resultCard.style.opacity = 1, 10);
        }
    </script>
</body>
</html>
'''


# Routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']

        # Process text
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(accelerator.device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        predicted_idx = np.argmax(probs)

        return jsonify({
            "emotion": emotion_labels[predicted_idx].upper(),
            "confidence": f"{probs[predicted_idx] * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# Run with ngrok
if __name__ == '__main__':
    # Cleanup previous runs
    shutil.rmtree('static', ignore_errors=True)
    os.makedirs('static', exist_ok=True)

    # Start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print(f" * Professional Interface: {public_url}")

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
