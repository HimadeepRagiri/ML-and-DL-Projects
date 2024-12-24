import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .config import CONFIG, DEVICE
from .environment import create_env


def setup_directories():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["video_dir"], exist_ok=True)


def create_model(env):
    return PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        seed=CONFIG["seed"],
        device=DEVICE
    )


def train_model(model, env):
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="ppo_carracing_checkpoint"
    )

    print("Starting training...")
    model.learn(
        total_timesteps=CONFIG["timesteps"],
        callback=checkpoint_callback
    )
    print("Training complete.")

    model.save(CONFIG["save_model_path"])