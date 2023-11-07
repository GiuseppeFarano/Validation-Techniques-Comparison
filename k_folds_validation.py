from sklearn.model_selection import KFold
from utilities import normalization, evaluation_kf
from sklearn.linear_model import LogisticRegression


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

        [x_train, x_test] = normalization(x_train, x_test)
        logistic = LogisticRegression()
        logistic.fit(x_train, y_train)
        y_pred = logistic.predict(x_test)

        print("\nClassification metrics with K-fold Cross validation:" + f" Fold{i+1}")
        [total_acc, total_prec, total_rec, total_f1] = evaluation_kf(y_test, y_pred, total_acc, total_prec, total_rec, total_f1)
    total_acc = total_acc/5
    total_prec = total_prec/5
    total_rec = total_rec/5
    total_f1 = total_f1/5
    print(f'\nTotal metrics from k-folds: Accuracy {total_acc}', f'Precision {total_prec}', f'Recall {total_rec}', f'F1 score {total_f1}')
