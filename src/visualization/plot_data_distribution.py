import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dataset_distribution():
    sns.set_theme(style="darkgrid")

    # Plot distribution for training and internal test datasets
    data = {"Normal": ["foo", "foo"], "Pneumonia": ["foo", "foo"]}
    index = ["Train Set", "Internal Test (Validation) Set"]
    df = pd.DataFrame(data=data, index=index)

    df.plot(kind='bar', stacked=True, color=['lightblue', 'darkblue'])
    plt.xlabel('Dataset Partitions', labelpad=20)
    plt.ylabel('Number of Chest Radiographs', labelpad=20)
    plt.xticks(rotation=0, fontsize=9.5)
    plt.yticks(fontsize=9.5)
    plt.title("Distribution of Training and Validation Datasets")
    plt.show()

    # Plot distribution for external test dataset
    data = {"Normal": ["foo"], "Pneumonia": ["foo"]}
    index = ["External Test Set"]
    df = pd.DataFrame(data=data, index=index)

    df.plot(kind='bar', stacked=True, color=['lightblue', 'darkblue'])
    plt.xlabel('Dataset Partitions', labelpad=20)
    plt.ylabel('Number of Chest Radiographs', labelpad=20)
    plt.xticks(rotation=0, fontsize=9.5)
    plt.yticks(fontsize=9.5)
    plt.title("Distribution of External Test Dataset")
    plt.show()
