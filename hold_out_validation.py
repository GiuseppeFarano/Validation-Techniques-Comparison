from utilities import normalization, evaluation
from sklearn.linear_model import LogisticRegression


def h_o_validation(x, y):
    # Split training-data and test-data: HOLD-OUT
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
    evaluation(y_test, y_pred)
