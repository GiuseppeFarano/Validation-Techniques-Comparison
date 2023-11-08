import numpy as np
import pandas as pd
np.random.seed(123)

# Import functions to compute different types of validations and to plot their performance
from hold_out_validation import validation_hold_out
from stratification_validation import validation_stratification
from k_fold_validation import validation_k_fold
from random_sub_sampling_validation import validation_random_sub_sampling
from utilities import plot_performance

# Import data and select features and target
diabetes = pd.read_csv('datasets/diabetes.csv')
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = diabetes[selected_features].values
y = diabetes['Outcome'].values

# Validation with different techniques
techniques = ('Hold-out', 'Stratification', 'K-folds', 'Random Sub-Sampling')
performances_hold_out = validation_hold_out(x, y)
performances_stratification = validation_stratification(diabetes, selected_features)
performances_k_fold = validation_k_fold(x, y)
performances_random_sub_sampling = validation_random_sub_sampling(x, y)
performances = (performances_hold_out, performances_stratification, performances_k_fold, performances_random_sub_sampling)

# Plot the obtained metrics
plot_performance(techniques, performances)
