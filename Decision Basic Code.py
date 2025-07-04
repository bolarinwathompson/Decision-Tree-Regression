# Regression Tree - Basic Templates

# Import Required Python Packages
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# Import Sample Data
my_df = pd.read_csv("sample_data_regression.csv")

# Split the Data into input and Output Data
X = my_df.drop(["output"], axis=1)
y = my_df["output"]

# Split the Data into Test and Train Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate our Model Object
regressor = DecisionTreeRegressor(min_samples_leaf = 7)

# Train our Model
regressor.fit(X_train, y_train)

# Assess Model Accuracy on Test Data
y_pred_test = regressor.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)

# Assess Model Accuracy on Training Data (for Overfitting Check)
y_pred_train = regressor.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)

# Print R-squared Scores
print(f"R-squared on Test Data: {r2_test:.4f}")
print(f"R-squared on Training Data: {r2_train:.4f}")

# Plot the Decision Tree

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 15))
tree = plot_tree(regressor,
                 feature_names=X.columns,
                 filled=True,
                 rounded=True,
                 fontsize=24)



