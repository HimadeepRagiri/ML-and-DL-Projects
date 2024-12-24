import gymnasium as gym
from stable_baselines3 import PPO
import cv2
import numpy as np


def visualize_agent(model_path, video_path="agent_gameplay.mp4", num_episodes=3, fps=60):
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    model = PPO.load(model_path)

    desired_width = 800
    desired_height = 600

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (desired_width, desired_height))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (desired_width, desired_height),
                               interpolation=cv2.INTER_LANCZOS4)

            cv2.putText(frame, f"Episode: {episode + 1}", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Reward: {total_reward:.1f}", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        print(f"Episode {episode + 1} completed with reward: {total_reward:.2f}")

    out.release()
    env.close()