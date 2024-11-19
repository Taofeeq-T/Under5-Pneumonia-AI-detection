import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_metrics(initial_history_path, fine_tune_history_path):
    sns.set_theme(style="darkgrid")

    # Load training and fine-tuning histories
    history = pd.read_csv(initial_history_path)
    fine_history = pd.read_csv(fine_tune_history_path)

    metrics = ['accuracy', 'loss', 'f1_score', 'precision', 'recall', 'auc']
    for metric in metrics:
        history[metric] = history.get(metric, pd.Series())
        fine_history[metric] = fine_history.get(metric, pd.Series())

    combined_metrics = {m: pd.concat([history[m], fine_history[m]], ignore_index=True) for m in metrics}

    # Plotting all metrics
    plt.figure(figsize=(18, 18))
    titles = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        plt.plot(combined_metrics[metric], label=f'Training {metric.capitalize()}', c="darkblue", alpha=0.85)
        plt.plot(combined_metrics[f'val_{metric}'], label=f'Validation {metric.capitalize()}', c="darkorange", alpha=0.85)
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'Training and Validation {titles[i-1]}')
    plt.tight_layout()
    plt.show()
