import os 
import matplotlib.pyplot as plt

def plot_predictions(actual, preds, n=200, title="Naive Baseline vs Actual"):
    plt.figure(figsize=(12,4))
    plt.plot(actual[:n], label="Actual", color="orange")
    plt.plot(preds[:n], label="Naive Prediction", linestyle="--", color="blue")
    plt.xlabel("Validation sample index")
    plt.ylabel("GPU Power (W)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists("res_plots"):
        os.makedirs("res_plots")
    plt.savefig("res_plots/naive_model_loss.png")
