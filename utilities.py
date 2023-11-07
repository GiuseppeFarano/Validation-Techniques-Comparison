import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def normalization(X_train, X_test):
    mean = X_train.mean(axis=0)  # ONLY ON TRAINING SAMPLES
    std = X_train.std(axis=0)  # ONLY ON TRAINING SAMPLES
    X_train = (X_train - mean) / std  # both on training set and test set
    X_test = (X_test - mean) / std  # both on training set and test set
    X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))  # Add a column of ones for the bias term
    X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
    return X_train, X_test


def evaluation(y_test, y_pred):
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred)
    rec_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    print(f'Accuracy {acc_test}', f'Precision {prec_test}', f'Recall {rec_test}', f'F1 score {f1_test}')


def evaluation_kf(y_test, y_pred, total_acc, total_prec, total_rec, total_f1):
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred)
    rec_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    total_acc = total_acc + acc_test
    total_prec = total_prec + prec_test
    total_rec = total_rec + rec_test
    total_f1 = total_f1 + f1_test
    print(f'Accuracy {acc_test}', f'Precision {prec_test}', f'Recall {rec_test}', f'F1 score {f1_test}')
    return total_acc, total_prec, total_rec, total_f1
