import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from .config import CONFIG

def create_env(env_name, monitor_path=None):
    env = gym.make(env_name, render_mode="rgb_array")
    env.reset(seed=CONFIG["seed"])
    if monitor_path:
        env = Monitor(env, monitor_path)
    return DummyVecEnv([lambda: env])