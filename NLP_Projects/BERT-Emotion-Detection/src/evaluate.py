from accelerate import Accelerator
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import Config


def evaluate_model(test_loader, class_weights):
    """Evaluate the trained model on the test set"""
    accelerator = Accelerator()

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS
    )

    # Prepare components
    model, test_loader = accelerator.prepare(model, test_loader)

    # Load checkpoint
    accelerator.load_state(os.path.join(Config.CHECKPOINT_DIR, "best_model"))

    # Initialize loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(accelerator.device))

    # Evaluation
    model.eval()
    test_loss = 0
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            test_loss += loss.item()

            preds.append(accelerator.gather(outputs.logits.argmax(dim=1)))
            true_labels.append(accelerator.gather(batch["labels"]))

    preds = torch.cat(preds).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()

    print("\n===== Final Test Results =====")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy_score(true_labels, preds):.4f}")
    print(f"Macro F1: {f1_score(true_labels, preds, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=Config.LABEL_NAMES))