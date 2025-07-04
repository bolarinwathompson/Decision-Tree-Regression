# Advanced Classification Tree - ABC Grocery Task 



# Import Required Packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np 

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder


# -------------------------------------------------------------------
# Import Sample Data
data_for_model = pickle.load(open("abc_classification_modelling.p", "rb"))

# -------------------------------------------------------------------
# Drop Unnecessary Columns
data_for_model.drop("customer_id", axis=1, inplace=True)

# -------------------------------------------------------------------
# Shuffle the Data
data_for_model = shuffle(data_for_model, random_state=42)

# Class Balance
data_for_model["signup_flag"].value_counts(normalize=True)

# -------------------------------------------------------------------
# Deal with Missing Data
print("Missing values per column:")
print(data_for_model.isna().sum())
data_for_model.dropna(how="any", inplace=True)


# -------------------------------------------------------------------
# Split Input Variables and Output Variable
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]


# -------------------------------------------------------------------
# Split out Training and Test Sets (this step ensures X_test is defined)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# -------------------------------------------------------------------
# Feature Selection: One-Hot Encoding for Categorical Variable(s)
categorical_vars = ["gender"]

# Use sparse_output=False for scikit-learn 1.2+; change to sparse=False if using an earlier version.
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Fit the encoder on the training set's categorical column and transform both training and test sets.
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the encoded feature names using the new method.
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert the encoded arrays into DataFrames (keeping the original indices for proper alignment).
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder_feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder_feature_names, index=X_test.index)

# Remove the original categorical column(s) from X_train and X_test.
X_train_clean = X_train.drop(columns=categorical_vars)
X_test_clean = X_test.drop(columns=categorical_vars)

# Concatenate the cleaned DataFrames with their corresponding encoded DataFrames.
X_train = pd.concat([X_train_clean, X_train_encoded_df], axis=1)
X_test = pd.concat([X_test_clean, X_test_encoded_df], axis=1)


# Model Assessment 
clf = DecisionTreeClassifier(random_state = 42, max_depth = 5)
clf.fit(X_train, y_train)


# Predict The Test Score
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

import numpy as np 
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix): 
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()


# Accuracy (the number of correct clasification out of all attempted classifications)
accuracy_score(y_test, y_pred_class)

# Precision of all(of all observations that were predicted as positive, how many were actually posiitve)
precision_score(y_test, y_pred_class)

# Recall (Of all Positive observations, how many did we predict as positive)
recall_score(y_test, y_pred_class)

# Fi-Score(the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)


# Finding the best max_depth
max_depth_list = list(range(1,15))
accuracy_scores = []

for depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]


# Plot the max depth
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker="x", color="red")
plt.title(f"Accuracy (F1 Score) by Max Depth\nOptimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)})")
plt.ylabel("Accuracy(F1 Score)")
plt.xlabel("Max Depth")
plt.tight_layout()
plt.show()


# Plot our model
plt.figure(figsize=(25,15))
plot_tree(clf,
          feature_names = X_train.columns,
          filled = True,
          rounded = True,
          fontsize = 16)
plt.show()