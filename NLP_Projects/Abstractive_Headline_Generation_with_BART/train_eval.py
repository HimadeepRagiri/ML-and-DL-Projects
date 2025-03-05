from transformers import Trainer, TrainingArguments
from config import Config

def setup_trainer(model, train_tokenized, val_tokenized, data_collator, tokenizer):
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        per_device_train_batch_size=Config.BATCH_SIZE,
        evaluation_strategy="no",
        logging_steps=100,
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.TRAIN_EPOCHS,
        save_total_limit=1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer

def train_model(trainer):
    print("Starting training...")
    trainer.train()
    return trainer

def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    return eval_results
