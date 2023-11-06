import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def h_o_validation(x, y):
    # Split training-data and test-data: HOLD-OUT
    train_ratio = 0.8
    train_size = int(len(x) * train_ratio)
    X_train = x[:train_size]
    y_train = y[:train_size]
    X_test = x[train_size:]
    y_test = y[train_size:]

    # Normalization
    mean = X_train.mean(axis=0)  # ONLY ON TRAINING SAMPLES
    std = X_train.std(axis=0)  # ONLY ON TRAINING SAMPLES
    X_train = (X_train - mean) / std  # both on training set and test set
    X_test = (X_test - mean) / std  # both on training set and test set

    # Add a column of ones for the bias term REMEMBER TO ADD AFTER NORMALIZATION
    X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
    X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

    # Training, predicting and evaluating with mini-batch
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred_mb = logistic.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_mb)
    prec_test = precision_score(y_test, y_pred_mb)
    rec_test = recall_score(y_test, y_pred_mb)
    f1_test = f1_score(y_test, y_pred_mb)

    #print metrics
    print("\nClassification metrics with Hold-out validation:")
    print(f'Accuracy {acc_test}', f'Precision {prec_test}', f'Recall {rec_test}', f'F1 score {f1_test}')
