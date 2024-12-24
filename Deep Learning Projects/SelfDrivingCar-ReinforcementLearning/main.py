from src.config import CONFIG
from src.environment import create_env
from src.training import setup_directories, create_model, train_model
from src.evaluation import evaluate_model
from src.visualization import visualize_agent
from stable_baselines3 import PPO


def main():
    # Setup
    setup_directories()
    env = create_env(CONFIG["env_name"])

    # Training
    model = create_model(env)
    train_model(model, env)

    # Evaluation
    print("Final evaluation...")
    final_reward = evaluate_model(model, env, CONFIG["evaluate_episodes"])
    print(f"Final evaluation completed. Average reward: {final_reward}")

    # Visualization
    print("Creating visualization...")
    visualize_agent(CONFIG["save_model_path"])


if __name__ == "__main__":
    main()