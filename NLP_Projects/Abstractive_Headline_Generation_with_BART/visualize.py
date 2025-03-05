import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_loss(trainer):
    if trainer.state.log_history:
        logs = trainer.state.log_history
        df_logs = pd.DataFrame(logs)
        if "loss" in df_logs.columns:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_logs, x="step", y="loss")
            plt.title("Training Loss Over Steps")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.show()
        else:
            print("No training loss data available for visualization.")
    else:
        print("No training log history available.")
        