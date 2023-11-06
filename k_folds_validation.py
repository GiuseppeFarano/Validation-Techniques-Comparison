import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def kf_validation(x, y):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    total_acc = 0
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        mean = x_train.mean(axis=0)  # Normalization
        std = x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))  # Add a column of ones for the bias term
        x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))
        logistic = LogisticRegression()
        logistic.fit(x_train, y_train)
        y_pred_mb = logistic.predict(x_test)
        acc_test = accuracy_score(y_test, y_pred_mb)
        prec_test = precision_score(y_test, y_pred_mb)
        rec_test = recall_score(y_test, y_pred_mb)
        f1_test = f1_score(y_test, y_pred_mb)
        total_acc = total_acc + acc_test
        total_prec = total_prec + prec_test
        total_rec = total_rec + rec_test
        total_f1 = total_f1 + f1_test
        print("\nClassification metrics with K-fold Cross validation:" + f"Fold{i+1}")
        print(f'Accuracy {acc_test}', f'Precision {prec_test}', f'Recall {rec_test}', f'F1 score {f1_test}')
    total_acc = total_acc/5
    total_prec = total_prec/5
    total_rec = total_rec/5
    total_f1 = total_f1/5
    print(f'\nTotal metrics from k-folds: Accuracy {total_acc}', f'Precision {total_prec}', f'Recall {total_rec}', f'F1 score {total_f1}')
