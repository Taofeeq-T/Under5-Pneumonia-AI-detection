import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(test_labels, base_predictions, fine_predictions, class_names):
    cm_base = confusion_matrix(test_labels, base_predictions)
    cm_fine = confusion_matrix(test_labels, fine_predictions)

    plt.figure(figsize=(18, 9))

    # Base model confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_base, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r', 
                yticklabels=class_names, xticklabels=class_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: Base Model')

    # Fine-tuned model confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_fine, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r', 
                yticklabels=class_names, xticklabels=class_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: Fine-Tuned Model')

    plt.tight_layout()
    plt.show()
