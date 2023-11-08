import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

'''
Module containing function to normalize data, evaluate models performance and plot them
'''

def normalization(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))  # Add a column of ones for the bias term
    X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
    return X_train, X_test


def evaluation(y_test, y_pred):
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred)
    rec_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    print(f'Accuracy {acc_test}', f'Precision {prec_test}', f'Recall {rec_test}', f'F1 score {f1_test}')
    return acc_test, prec_test, rec_test, f1_test


def evaluation_kf(y_test, y_pred, total_acc, total_prec, total_rec, total_f1):
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred)
    rec_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    total_acc = total_acc + acc_test
    total_prec = total_prec + prec_test
    total_rec = total_rec + rec_test
    total_f1 = total_f1 + f1_test
    return total_acc, total_prec, total_rec, total_f1


def plot_performance(techniques, performances):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    num_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Creation of 4 separate plots
    fig.suptitle("Performance Metrics Comparison")

    # Make 4 plots, each one for a metric
    for i in range(num_metrics):
        row, col = divmod(i, 2)  # Compute the position of the plots
        ax = axes[row, col]
        x = np.arange(len(techniques))
        width = 0.3  # Bars width

        # Plot the metric value of each validation technique
        ax.bar(x[0], performances[0][i], width, color='skyblue')
        ax.bar(x[1], performances[1][i], width, color='lightcoral')
        ax.bar(x[2], performances[2][i], width, color='purple')
        ax.bar(x[3], performances[3][i], width, color='orange')

        # Manage scale, ticks, labels and title of the plots
        ax.set_xticks(x)
        ax.set_xticklabels(techniques)
        ax.set_title(f'{metrics[i]} Comparison')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value")

        # Grid
        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_yticks(np.arange(0, 1, 0.05))

    plt.tight_layout()
    plt.show()
