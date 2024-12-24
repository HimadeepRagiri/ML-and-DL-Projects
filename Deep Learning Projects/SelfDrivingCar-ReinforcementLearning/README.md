# Self-Driving Car using Reinforcement Learning

A deep reinforcement learning project that trains an autonomous agent to drive a car in a racing environment using the PPO (Proximal Policy Optimization) algorithm. This implementation uses the CarRacing-v3 environment from Gymnasium (formerly OpenAI Gym).

## Project Overview

This project demonstrates the application of deep reinforcement learning to autonomous driving in a simulated environment. The agent learns to navigate a race track by processing visual input and controlling the car's acceleration, braking, and steering through trial and error.

### Key Features
- Implementation of PPO algorithm for continuous control
- CNN-based policy for processing visual input
- Checkpoint-based training with automatic saves
- Performance evaluation metrics
- High-quality video generation of trained agent
- Detailed training visualization in Jupyter notebook

## Requirements

```bash
# Core Libraries
stable-baselines3[extra]
gymnasium[box2d]
pygame
moviepy
swig
torch
numpy
matplotlib
opencv-python
```

## Project Structure

```
SelfDrivingCar-ReinforcementLearning/
│
├── requirements.txt          # Project dependencies
├── README.md                # Project documentation
├── notebook/                # Jupyter notebooks
│   ├── SelfDrivingCar_ReinforcementLearning.ipynb  # Implementation notebook
├── src/                     # Source code
│   ├── config.py           # Configuration parameters
│   ├── environment.py      # Environment setup
│   ├── training.py         # Training functions
│   ├── evaluation.py       # Evaluation metrics
│   ├── visualization.py    # Visualization utilities
├── ppo_carracing_model     # Trained model
├── videos/                 # Generated videos
│   ├── agent_gameplay      # Agent performance recordings
└── main.py                 # Main execution script
```

## Installation

1. Clone the repository

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training a New Model

Run the main script to start training:
```bash
python main.py
```

This will:
1. Set up the CarRacing environment
2. Initialize the PPO model with CNN policy
3. Train for 500,000 timesteps
4. Save checkpoints every 10,000 steps
5. Evaluate the model's performance
6. Generate gameplay videos

### Using the Pre-trained Model

To use the pre-trained model for visualization:
```python
from src.visualization import visualize_agent

visualize_agent("ppo_carracing_model", "videos/agent_gameplay.mp4")
```

## Implementation Details

### Environment
- **Gymnasium Environment**: CarRacing-v3
- **Observation Space**: RGB images (96x96x3)
- **Action Space**: Continuous (3 dimensions)
  - Steering: [-1, 1]
  - Gas: [0, 1]
  - Brake: [0, 1]

### Model Architecture
- **Policy**: CnnPolicy (Stable-Baselines3)
- **Architecture Type**: Actor-Critic
- **Neural Network**:
  - CNN layers for visual feature extraction
  - Fully connected layers for action prediction
  - Value function estimation for advantage computation

### Training Configuration
- **Total Timesteps**: 500,000
- **Learning Rate**: Default PPO learning rate
- **Batch Size**: Default PPO batch size
- **Checkpoint Frequency**: Every 10,000 steps
- **Device**: CUDA if available, else CPU

## Results and Performance

### Training Metrics
- Average reward over evaluation episodes: 250
- Successful track completion rate: High
- Stable learning curve with consistent improvement

### Visualization
The project includes high-quality video recordings of the trained agent's performance:
- 800x600 resolution
- 60 FPS
- On-screen metrics display
- Multiple episodes showcase

## Jupyter Notebook

The `SelfDrivingCar_ReinforcementLearning.ipynb` notebook provides:
- Detailed implementation walkthrough
- Training process visualization
- Performance analysis
- Results interpretation

## Customization

You can modify the training parameters in `src/config.py`:
```python
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
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Gymnasium team for the CarRacing environment
- Stable-Baselines3 team for the PPO implementation
- Original PPO paper authors (Schulman et al.)

## Contact

For questions or feedback, please open an issue in the GitHub repository.

---
**Note**: This project is for educational purposes and demonstrates the application of reinforcement learning to autonomous driving in a simulated environment.