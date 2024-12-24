import torch

CONFIG = {
    "env_name": "CarRacing-v3",
    "timesteps": 500_000,
    "checkpoint_dir": "./checkpoints/",
    "video_dir": "./videos/",
    "save_model_path": "./ppo_carracing_model.zip",
    "evaluate_episodes": 5,
    "video_length": 1000,
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")