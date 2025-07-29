import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# load your two‐feature dataset
velocities_df = pd.read_csv("final_aggregated_data_TwoFeature.csv")

# the two features you want to use:
features = ['velocity_z', 'deviation']

# for neat formatting later
def print_separator():
    print("\n" + "-"*60 + "\n")

# loop over every distinct task
for task in velocities_df['task'].unique():
    df = velocities_df[velocities_df['task'] == task]
    X = df[features]
    y = df['label']
    
    # split (80:20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, 
        stratify=y, random_state=42
    )
    
    # train
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # predict & evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print_separator()
    print(f"Task = {task!r}")
    print(f"  → Accuracy: {acc:.3f}")
    print("  → Classification report:")
    print(classification_report(y_test, y_pred, target_names=['Unintended','Intentional']))
    
    # # confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print("  → Confusion matrix:")
    # print(pd.DataFrame(cm,
    #                    index=['Actual U','Actual I'],
    #                    columns=['Pred U','Pred I']))
    
    # fitted Gaussian parameters
    theta = pd.DataFrame(model.theta_,
                         columns=features,
                         index=['Unintended','Intentional'])
    sigma = pd.DataFrame(np.sqrt(model.var_),
                         columns=features,
                         index=['Unintended','Intentional'])
    priors = pd.Series(model.class_prior_,
                       index=['Unintended','Intentional'])
    
    print("\n  Means (θ):")
    print(theta.round(4))
    print("\n  Std Deviations (σ):")
    print(sigma.round(4))
    print("\n  Priors:")
    print(priors.round(4))

print_separator()
