from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm


#Load our agregated velocity data which is a csv file
velocities_df = pd.read_csv("final_aggregated_velocity_data.csv")

#display some basic info
print(velocities_df.info())
print(velocities_df.head())
print(velocities_df['label'].value_counts())    

X = velocities_df[['velocity_z']] # here we use double parathesis to make it a 2D array, which is required by the fit method
Y = velocities_df['label'] # Y is 1D with elements 0 or 1

#optional feature scaling step, we could investigate if it improves the model
#scaler = StandardScaler()
#X = scaler.fit_transform(X)


#Split the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 40, stratify = Y)

#Train the model
model = GaussianNB()
model.fit(X_train, Y_train)


#Predict the test set
Y_pred = model.predict(X_test)

#Evaluate the model
print("Accuracy score:", accuracy_score(Y_test, Y_pred))
print("Classification report:")
print(classification_report(Y_test, Y_pred))
#confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
# Extract the means and variances (used to compute the Gaussian distributions)
print("\nMeans of Gaussians for each class (Unintended=0, Intended=1):")
print(model.theta_)

print("\nStandard deviations of Gaussians for each class (Unintended=0, Intended=1):")
print(np.sqrt(model.var_))
# Filter unintended samples that were misclassified as intended
misclassified_indices = (Y_test == 1) & (Y_pred == 0)

# Extract the indices of misclassified samples
misclassified_indices_full = X_test[misclassified_indices].index

# Retrieve the corresponding rows from the original DataFrame
misclassified_samples = velocities_df.loc[misclassified_indices_full]

# Print the count of misclassified samples
print(f"\nNumber of misclassified intended samples: {len(misclassified_samples)}")

# Loop through each misclassified sample and print its details
print("\nMisclassified intended samples:")
for idx, row in misclassified_samples.iterrows():
    print(row.to_dict())


# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Unintended', 'Intended'],
#             yticklabels=['Unintended', 'Intended'])
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.title("confusion matrix")
# plt.show()# Plot the data with two fitted Gaussian distributions
# plt.figure(figsize=(10, 7))

# Extract the means and standard deviations for each class
mean_unintended, mean_intended = model.theta_.flatten()
std_unintended, std_intended = np.sqrt(model.var_.flatten())

# Generate points for the Gaussian curves
x = np.linspace(X['velocity_z'].min(), X['velocity_z'].max(), 1000)
plt.figure(figsize=(8, 5), dpi=100)
# Plot the histogram of data
sns.histplot(X[Y == 0]['velocity_z'], bins=30, kde=False, color='blue', alpha=0.5, label='Unintended')
sns.histplot(X[Y == 1]['velocity_z'], bins=30, kde=False, color='red', alpha=0.5, label='Intended')

# Correct the Gaussian scaling
bin_width = (X['velocity_z'].max() - X['velocity_z'].min()) / 30  # Bin width
scaling_unintended = len(X[Y == 0]) * bin_width
scaling_intended = len(X[Y == 1]) * bin_width

# Plot the Gaussian fits
plt.plot(x, norm.pdf(x, mean_unintended, std_unintended) * scaling_unintended, color='blue', label='Fitted Gaussian (Unintended)')
plt.plot(x, norm.pdf(x, mean_intended, std_intended) * scaling_intended, color='red', label='Fitted Gaussian (Intended)')

# Add labels and legend
#plt.title("Data Distribution with Fitted Gaussian Curves")
plt.xlabel("Z-axis velocity (cm/s)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.legend(fontsize=13)
from matplotlib.ticker import FuncFormatter
def times_100(x, _):
    return f"{x * 100:.0f}"
plt.gca().xaxis.set_major_formatter(FuncFormatter(times_100))
plt.tight_layout()
plt.savefig("gaussian_velocity_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
