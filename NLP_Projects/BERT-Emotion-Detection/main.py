import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from config import Config
from data_preprocessing import load_and_preprocess_data
from model import initialize_model
from train import train_model
from evaluate import evaluate_model

# Create directories
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)

# Load and preprocess data
dataset, train_loader, val_loader, test_loader, class_weights, tokenizer = load_and_preprocess_data()

# Data exploration
print("\n===== Dataset Structure =====")
print(dataset)

print("\n===== Label Distribution =====")
label_counts = Counter(dataset["train"][Config.LABEL_FIELD])
print({Config.LABEL_NAMES[k]: v for k, v in label_counts.items()})

# Visualize class distribution
plt.figure(figsize=(10, 5))
plt.bar(Config.LABEL_NAMES, [label_counts[i] for i in range(6)])
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.show()

# Text length analysis
text_lengths = [len(text.split()) for text in dataset["train"][Config.TEXT_FIELD]]
print(f"\nAverage text length: {np.mean(text_lengths):.1f} words")
print(f"95th percentile length: {np.percentile(text_lengths, 95)} words")

# Initialize and train model
model = initialize_model()
train_model(model, train_loader, val_loader, class_weights)

# Evaluate
evaluate_model(test_loader, class_weights)


# Prediction function
def predict(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH,
        return_tensors="pt"
    )

    accelerator = Accelerator()
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS
    )
    model = accelerator.prepare(model)
    accelerator.load_state(os.path.join(Config.CHECKPOINT_DIR, "best_model"))

    with torch.no_grad():
        outputs = model(**inputs.to(accelerator.device))

    return Config.LABEL_NAMES[outputs.logits.argmax().item()]


# Test examples
print("\nTest Predictions:")
print("Text: 'I feel so happy today!' →", predict("I feel so happy today!"))
print("Text: 'This situation terrifies me' →", predict("This situation terrifies me"))