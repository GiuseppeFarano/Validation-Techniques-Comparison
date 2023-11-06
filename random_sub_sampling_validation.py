import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def random_validation(diabetes, selected_features):
    validation_ratio = 0.2
    train_indices = []
    valid_indices = []
    indices = diabetes.index
    total_number = diabetes.shape[0]
    validation_size = int(total_number*validation_ratio)
    chosen_random_indexes = np.random.choice(indices, validation_size, replace=True)
    valid_indices.extend(chosen_random_indexes)
    train_indices = list(set(indices) - set(chosen_random_indexes))

    train_set = diabetes.loc[train_indices]  #mi prendo gli elementi corrispondenti agli indici selezionati
    valid_set = diabetes.loc[valid_indices]  #mi prendo gli elementi corrispondenti agli indici selezionati
    X_train = train_set[selected_features]
    X_test = valid_set[selected_features]
    y_train = train_set['Outcome']
    y_test = valid_set['Outcome']

    # Normalization
    train_set = train_set.drop("Outcome", axis=1)
    mean = train_set.mean(axis=0)
    std = train_set.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

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
    print("\nClassification metrics with Stratification validation:")
    print(f'Accuracy {acc_test}', f'Precision {prec_test}', f'Recall {rec_test}', f'F1 score {f1_test}')
