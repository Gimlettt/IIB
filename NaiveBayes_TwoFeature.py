from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.ticker as ticker

characteristic_one = 'velocity_z'
#characteristic_two = 'velocity_xy'
characteristic_two = 'deviation'
#Load our agregated velocity data which is a csv file
velocities_df = pd.read_csv("final_aggregated_data_TwoFeature.csv")

#display some basic info
print(velocities_df.info())
print(velocities_df.head())
print(velocities_df['label'].value_counts())    

X = velocities_df[[characteristic_one,characteristic_two]] # here we use double parathesis to make it a 2D array, which is required by the fit method
Y = velocities_df['label'] # Y is 1D with elements 0 or 1

# #optional feature scaling step
# scaler = StandardScaler()
# X = scaler.fit_transform(X)


#Split the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

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
print("\nMeans of Gaussians (feature-wise) for each class (Unintended=0, Intended=1):")
print(pd.DataFrame(model.theta_, columns=[characteristic_one, characteristic_two], index=['Unintended', 'Intended']))

print("\nStandard deviations of Gaussians (feature-wise) for each class (Unintended=0, Intended=1):")
print(pd.DataFrame(np.sqrt(model.var_), columns=[characteristic_one, characteristic_two], index=['Unintended', 'Intended']))

print("\nPrior probabilities for each class (Unintended=0, Intended=1):")
priors = pd.Series(model.class_prior_, index=['Unintended', 'Intended'])
print(priors)

correlation = velocities_df[[characteristic_one, characteristic_two]].corr()
print("\nFeature Correlation:\n", correlation)

# # Filter unintended samples that were misclassified as intended
# misclassified_indices = (Y_test == 1) & (Y_pred == 0)

# # Extract the indices of misclassified samples
# misclassified_indices_full = X_test[misclassified_indices].index

# # Retrieve the corresponding rows from the original DataFrame
# misclassified_samples = velocities_df.loc[misclassified_indices_full]

# # Print the count of misclassified samples
# print(f"\nNumber of misclassified intended samples: {len(misclassified_samples)}")

# # Loop through each misclassified sample and print its details
# print("\nMisclassified intended samples:")
# for idx, row in misclassified_samples.iterrows():
#     print(row.to_dict())



plt.figure(figsize=(8 ,5), dpi=100)

# Assume the second feature is stored in the 'deviation' column.
# Create a new DataFrame for the second feature.
X2 = velocities_df[['deviation']]

# Compute the mean and standard deviation for the 'deviation' feature for each class.
# (These are computed directly from the data.)
mean_deviation_unintended = X2[Y == 0]['deviation'].mean()
mean_deviation_intended   = X2[Y == 1]['deviation'].mean()
std_deviation_unintended  = X2[Y == 0]['deviation'].std()
std_deviation_intended    = X2[Y == 1]['deviation'].std()

# Generate a range of x values spanning the range of the 'deviation' feature.
x2 = np.linspace(X2['deviation'].min(), X2['deviation'].max(), 1000)

# Plot the histogram of the 'deviation' feature for each class using the raw counts.
sns.histplot(X2[Y == 0]['deviation'], bins=30, kde=False, color='blue', alpha=0.5, label='Unintended')
sns.histplot(X2[Y == 1]['deviation'], bins=30, kde=False, color='red', alpha=0.5, label='Intended')

# Instead of matching the histogram's scaling via bin width, we simply multiply the Gaussian PDF values by 100.
plt.plot(
    x2,
    norm.pdf(x2, mean_deviation_unintended, std_deviation_unintended) ,
    color='blue'
    # label='Fitted Gaussian (Unintended)'
)
plt.plot(
    x2,
    norm.pdf(x2, mean_deviation_intended, std_deviation_intended) ,
    color='red'
    # label='Fitted Gaussian (Intended)'
)

#plt.title("Deviation Distribution with Fitted Gaussian Curves (×100)")
plt.xlabel("Deviation (cm)", fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.legend()
ax = plt.gca()
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: "0" if np.isclose(x, 0) else f"{-x * 100:.0f}")
)
ax.invert_xaxis()  # Flips the x-axis direction

plt.tight_layout()
plt.savefig("gaussian_deviation_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()