import pandas as pd
import numpy as np
####
# This code is for finding the error rate of each task
####
# List of file paths for the uploaded CSV files
file_paths = [
    'sketch_frame/Sketching_label_001.csv',
    'sketch_frame/Sketching_label_002.csv',
    'sketch_frame/Sketching_label_003.csv',
    'sketch_frame/Sketching_label_004.csv',
    'sketch_frame/Sketching_label_005.csv',
    'sketch_frame/Sketching_label_006.csv',
    'sketch_frame/Sketching_label_007.csv',
    'sketch_frame/Sketching_label_008.csv'
]



# Initialize a list to store error rates for each person
error_rates = []

# Iterate through each file, calculate the error rate, and append to the list
for file_path in file_paths:
    df = pd.read_csv(file_path)
    if 'label' in df.columns:
        unintentional_count = (df['label'] == 0).sum()
        error_rate = unintentional_count / 3  # Assuming 35 drops per person
        error_rates.append(error_rate)

# Calculate mean and standard deviation of error rates
mean_error_rate = np.mean(error_rates)
std_error_rate = np.std(error_rates)

print(mean_error_rate, std_error_rate)
