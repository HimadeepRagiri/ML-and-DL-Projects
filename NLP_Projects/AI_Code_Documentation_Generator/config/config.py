import os

class Config:
    """Configuration class containing all parameters for the project."""
    def __init__(self):
        # Base directory for all saved files
        self.base_dir = "AI_Code_Documentation_Generator"

        # Directory structure
        self.checkpoints_dir = os.path.join(self.base_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.base_dir, "tensorboard_logs")
        self.model_dir = os.path.join(self.base_dir, "final_model")
        self.logs_dir = os.path.join(self.base_dir, "training_logs")

        # Model and Dataset Parameters
        self.model_name = "gpt2"
        self.dataset_name = "code_search_net"
        self.max_length = 512

        # Dataset size
        self.train_subset_size = 50000
        self.val_subset_size = 5000

        # Special tokens for documentation generation
        self.code_token = "<CODE>"
        self.doc_token = "<DOC>"
        self.sep_token = "<SEP>"

        # Training Hyperparameters
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 5e-5
        self.num_train_epochs = 2
        self.warmup_steps = 100
        self.logging_steps = 100
        self.save_steps = 500
        self.eval_steps = 500
        self.fp16 = True

        # LoRA Parameters
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.lora_target_modules = ["c_attn", "c_proj"]