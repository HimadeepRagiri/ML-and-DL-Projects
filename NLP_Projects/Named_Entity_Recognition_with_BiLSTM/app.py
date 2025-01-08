from pyngrok import ngrok
import torch
from flask import Flask, request, jsonify, render_template_string
import logging
from src.config import DEVICE
from src.utils import predict
from torchtext.data.utils import get_tokenizer

# Disable unnecessary warnings
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NER Model Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 100px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: none;
        }
        .token {
            display: inline-block;
            padding: 2px 5px;
            margin: 2px;
            border-radius: 3px;
        }
        .entity {
            font-weight: bold;
        }
        .PER { background-color: #FFB6C1; }
        .ORG { background-color: #98FB98; }
        .LOC { background-color: #87CEFA; }
        .MISC { background-color: #DDA0DD; }
        .O { background-color: transparent; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Named Entity Recognition</h1>
        <div class="input-group">
            <textarea id="input-text" placeholder="Enter text for NER analysis..."></textarea>
            <button onclick="analyze()">Analyze Text</button>
        </div>
        <div id="result"></div>
    </div>

    <script>
        function analyze() {
            const text = document.getElementById('input-text').value;
            const resultDiv = document.getElementById('result');

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text}),
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.style.display = 'block';
                let html = '<h3>Results:</h3>';

                data.results.forEach(item => {
                    const token = item[0];
                    const tag = item[1];
                    const entityClass = tag.includes('-') ? tag.split('-')[1] : tag;
                    html += `<span class="token ${entityClass}">${token} <sub>${tag}</sub></span>`;
                });

                resultDiv.innerHTML = html;
            })
            .catch(error => {
                console.error('Error:', error);
                resultDiv.innerHTML = 'Error processing request';
                resultDiv.style.display = 'block';
            });
        }
    </script>
</body>
</html>
'''


def predict_text(text, model, tokenizer, TEXT, TAGS):
    """
    Predict NER tags for input text using the loaded model
    """
    return predict(model, text, tokenizer, TEXT, TAGS, DEVICE)


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get input text
        text = request.json['text']

        # Make prediction using the global model and tokenizer
        results = predict_text(text, model, tokenizer, TEXT, TAGS)

        return jsonify({'results': results})
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500


def start_server(loaded_model, text_vocab, tags_vocab):
    """
    Start the Flask server with the provided model and vocabularies
    """
    global model, TEXT, TAGS, tokenizer

    # Set global variables
    model = loaded_model
    TEXT = text_vocab
    TAGS = tags_vocab
    tokenizer = get_tokenizer("basic_english")

    # Make sure model is in eval mode
    model.eval()

    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f"\nPublic URL: {public_url}")

    # Run the app
    app.run(port=5000)


if __name__ == '__main__':
    # This will only run if app.py is run directly
    # For proper usage, call start_server() from main.py with the required parameters
    print("Please run main.py instead of running app.py directly")