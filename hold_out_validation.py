from utilities import normalization, evaluation
from sklearn.linear_model import LogisticRegression


def validation_hold_out(x, y):
    # Split training-data and test-data
    train_ratio = 0.8
    train_size = int(len(x) * train_ratio)
    X_train = x[:train_size]
    y_train = y[:train_size]
    X_test = x[train_size:]
    y_test = y[train_size:]

    # Normalization
    [X_train, X_test] = normalization(X_train, X_test)

    # Training and predicting
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)

    # Evaluating and print metrics
    print("Classification metrics with Hold-out validation:")
    [acc_test, prec_test, rec_test, f1_test] = evaluation(y_test, y_pred)
    return acc_test, prec_test, rec_test, f1_test
