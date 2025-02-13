import math
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.tensorboard import SummaryWriter

def setup_training(config, model, tokenized_train, tokenized_val, tokenizer):
    """Set up the training configuration and trainer."""
    # Generate a unique run name using timestamp
    from datetime import datetime
    run_name = f"doc_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_args = TrainingArguments(
        output_dir=config.checkpoints_dir,
        run_name=run_name,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=0.01,
        logging_dir=config.logs_dir,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=config.fp16,
        save_total_limit=2,
        report_to="wandb"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    return trainer

def train_model(trainer, config):
    """Train the model and log metrics."""
    print("Starting training...")
    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    # Training
    train_result = trainer.train()

    # Log training metrics
    writer.add_scalar("Training/Loss", train_result.training_loss)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    writer.close()
    return train_result

def evaluate_model(trainer, writer):
    """Evaluate the model and log metrics."""
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)

    print(f"Evaluation Results:")
    print(f"Loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    writer.add_scalar("Evaluation/Loss", eval_loss)
    writer.add_scalar("Evaluation/Perplexity", perplexity)

    return eval_results
