import numpy as np
from utilities import normalization, evaluation
from sklearn.linear_model import LogisticRegression


def strat_validation(diabetes, selected_features):
    # Split training-data and test-data: STRATIFICATION
    class_counts = diabetes['Outcome'].value_counts()
    validation_ratio = 0.2  # Specifica la percentuale di dati da utilizzare per il validation set (ad esempio, 20%)
    train_indices = []
    valid_indices = []
    for class_label, count in class_counts.items():  # Per ciascuna classe, seleziona casualmente una percentuale di righe per il validation set
        class_indices = diabetes[diabetes['Outcome'] == class_label].index  # individuo gli indici del dataset in cui ho outcome di una certa classe
        valid_size = int(count * validation_ratio)  # calcolo il numero di samples che voglio nel validation set, quindi per ogni classe il 20% di campioni di quella classe
        valid_indices_class = np.random.choice(class_indices, valid_size, replace=False)  # Estrai valid_size indici casuali fra quelli individuati, per il validation set
        valid_indices.extend(valid_indices_class)  # indici che uso per selezionare i dati di validazione
        train_indices_class = list(set(class_indices) - set(valid_indices_class))
        train_indices.extend(train_indices_class)  # indici che uso per selezionare i dati di training
    train_set = diabetes.loc[train_indices]  #mi prendo gli elementi corrispondenti agli indici selezionati
    valid_set = diabetes.loc[valid_indices]  #mi prendo gli elementi corrispondenti agli indici selezionati
    X_train = train_set[selected_features].values
    X_test = valid_set[selected_features].values
    y_train = train_set['Outcome'].values
    y_test = valid_set['Outcome'].values

    # Normalization
    [X_train, X_test] = normalization(X_train, X_test)

    # Training and predicting
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)

    # Evaluating and print metrics
    print("\nClassification metrics with Stratification validation:")
    evaluation(y_test, y_pred)
