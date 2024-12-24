import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder

def evaluate_model(model, env, episodes):
    total_rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    return avg_reward

def record_video(env, model, video_length, video_path):
    env = VecVideoRecorder(
        env,
        video_path,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="ppo_carracing"
    )
    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()