import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from config import Config


def train_model(model, train_loader, val_loader, class_weights):
    """Train the model with validation and checkpointing"""
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Prepare components
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Scheduler
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(accelerator.device))

    best_val_loss = float("inf")
    writer = SummaryWriter(Config.LOG_DIR)

    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")

        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        preds, true_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch["labels"])
                val_loss += loss.item()

                preds.append(accelerator.gather(outputs.logits.argmax(dim=1)))
                true_labels.append(accelerator.gather(batch["labels"]))

        avg_val_loss = val_loss / len(val_loader)
        preds = torch.cat(preds).cpu().numpy()
        true_labels = torch.cat(true_labels).cpu().numpy()

        val_acc = accuracy_score(true_labels, preds)
        val_f1 = f1_score(true_labels, preds, average="macro")

        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            accelerator.save_state(output_dir=os.path.join(Config.CHECKPOINT_DIR, "best_model"))

    writer.close()