from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from config import Config

# Model Setup
model_name = "facebook/bart-base"

def setup_model():
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable parameters:", trainable_params)
    print("Total parameters:", total_params)

    return model, tokenizer

# Tokenization
def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(example['article'],
                             max_length=Config.MAX_INPUT_LENGTH,
                             truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['highlights'],
                           max_length=Config.MAX_TARGET_LENGTH,
                           truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def prepare_data_for_training(train_sample, val_sample, tokenizer):
    print("Tokenizing training dataset...")
    train_tokenized = train_sample.map(tokenize_function, batched=True, remove_columns=["article", "highlights"], fn_kwargs={"tokenizer": tokenizer})

    print("Preparing and tokenizing validation dataset...")
    val_tokenized = val_sample.map(tokenize_function, batched=True, remove_columns=["article", "highlights"], fn_kwargs={"tokenizer": tokenizer})

    train_tokenized.set_format(type="torch")
    val_tokenized.set_format(type="torch")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)  # Model set in Trainer
    return train_tokenized, val_tokenized, data_collator
