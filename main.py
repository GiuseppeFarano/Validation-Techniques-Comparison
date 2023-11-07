import numpy as np
import pandas as pd
np.random.seed(123)

from hold_out_validation import h_o_validation
from stratification_validation import strat_validation
from k_folds_validation import kf_validation
from random_sub_sampling_validation import random_sub_samp_validation

# Import data and select features and target
diabetes = pd.read_csv('datasets/diabetes.csv')
diabetes = diabetes.sample(frac=1).reset_index(drop=True)  # Shuffling all samples to avoid group bias
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = diabetes[selected_features].values  # LOOK THAT x IS A NUMPY MATRIX
y = diabetes['Outcome'].values  # LOOK THAT x IS A NUMPY ARRAY

# Validation with different techniques
h_o_validation(x, y)
strat_validation(diabetes, selected_features)
kf_validation(x, y)
random_sub_samp_validation(x, y)
