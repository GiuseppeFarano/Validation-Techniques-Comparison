import numpy as np
from utilities import normalization, evaluation
from sklearn.linear_model import LogisticRegression


def validation_stratification(diabetes, selected_features):
    # Split training-data and test-data
    class_counts = diabetes['Outcome'].value_counts()  # Count the occurrences of each class label in the 'Outcome' column
    validation_ratio = 0.2  # Set the percentage of data to use for the validation phase
    train_indices = []  # Initialize a list to store training indices
    valid_indices = []  # Initialize a list to store validation indices

    # Loop through each class label and its count
    for class_label, count in class_counts.items():
        class_indices = diabetes[diabetes['Outcome'] == class_label].index  # Get indices of data points with the current class label
        valid_size = int(count * validation_ratio)  # Calculate the number of data points for validation for that specific class (this instruction characterizes stratification)
        valid_indices_class = np.random.choice(class_indices, valid_size, replace=False)  # Randomly select validation indices for this class
        valid_indices.extend(valid_indices_class)  # Add the selected validation indices for this class to the validation indices
        train_indices_class = list(set(class_indices) - set(valid_indices_class))  # Compute training indices for this class
        train_indices.extend(train_indices_class)  # Add the training indices for this class to the training indices

    train_set = diabetes.loc[train_indices]  # Create the training dataset using the training indices
    valid_set = diabetes.loc[valid_indices]  # Create the validation dataset using the validation indices
    X_train = train_set[selected_features].values  # Extract features for the training set
    X_test = valid_set[selected_features].values  # Extract features for the validation set
    y_train = train_set['Outcome'].values  # Extract class labels for the training set
    y_test = valid_set['Outcome'].values  # Extract class labels for the validation set

    # Normalization
    [X_train, X_test] = normalization(X_train, X_test)

    # Training and predicting
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)

    # Evaluating and print metrics
    print("\nClassification metrics with Stratification validation:")
    [acc_test, prec_test, rec_test, f1_test] = evaluation(y_test, y_pred)
    return acc_test, prec_test, rec_test, f1_test
