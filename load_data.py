import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the DataFrame(containing the data)
data_df = pd.read_pickle("/Users/jerry/Desktop/AR/logged_train_data_preprocessed/train_data_df_DEPTH0")

# Load the labels
labels = np.load("/Users/jerry/Desktop/AR/logged_train_data_preprocessed/train_data_labels_DEPTH0.npy", allow_pickle=True)

# Check the data
print("Data shape:", data_df.shape) 
print("Labels shape:", labels.shape)  
#printing the first row of the data to see the 75 feature type and their value
first_row = data_df.iloc[0]
print("First row with feature names:")
for feature, value in first_row.items():
    print(f"{feature}: {value}")

#a function to plot the feature value by label
def plot_feature_by_label(data_df, labels, feature_index):
    # Get the feature name
    feature_name = data_df.columns[feature_index]

    # Get the values of the selected feature for all samples
    feature_values = data_df.iloc[:, feature_index].values

    # Create lists to hold the values for each blocks of 10 samples corresponding to each label
    samples_label_0 = []
    samples_label_1 = []
    #create lists to hold the data-frame original index for plotting x-axis
    indices_label_0 = []
    indices_label_1 = []

    
    for i in range(len(labels)):
        #get 10 values
        block_values = feature_values[i*10:(i+1)*10]
        
        # Separate the values based on the label
        if labels[i] == 0:
            samples_label_0.extend(block_values)
            indices_label_0.extend(range(i*10,(i+1)*10))
        else:
            samples_label_1.extend(block_values)
            indices_label_1.extend(range(i*10,(i+1)*10))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(indices_label_0, samples_label_0, color='blue', label='Label 0 (Unintended)', alpha=0.6)
    plt.scatter(indices_label_1, samples_label_1, color='red', label='Label 1 (Intended)', alpha=0.6)

    # Label the plot according to the feature name
    plt.title(f'Plot of {feature_name} for All Samples')
    plt.xlabel('Sample Index')
    plt.ylabel(f'{feature_name} Value')
    plt.legend()
    plt.show()

# Example usage: plot the first feature (index 0)
plot_feature_by_label(data_df, labels, feature_index=45)