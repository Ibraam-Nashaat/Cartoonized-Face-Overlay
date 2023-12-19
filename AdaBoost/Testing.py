import numpy as np
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from AdaBoost import AdaBoostClassifier

# Generate synthetic data
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test custom AdaBoost
adaboost_custom = AdaBoostClassifier(n_estimators=50)
adaboost_custom.fit(X_train, y_train)
y_pred_custom = adaboost_custom.predict(X_test)

# Train and test scikit-learn AdaBoost
adaboost_sklearn = SklearnAdaBoost(n_estimators=50)
adaboost_sklearn.fit(X_train, y_train)
y_pred_sklearn = adaboost_sklearn.predict(X_test)

# Compare precision, recall, and F1 score
precision_custom = precision_score(y_test, y_pred_custom)
recall_custom = recall_score(y_test, y_pred_custom)
f1_custom = f1_score(y_test, y_pred_custom)

precision_sklearn = precision_score(y_test, y_pred_sklearn)
recall_sklearn = recall_score(y_test, y_pred_sklearn)
f1_sklearn = f1_score(y_test, y_pred_sklearn)

# Print the results
print("Custom AdaBoost Metrics:")
print(f"Precision: {precision_custom}")
print(f"Recall: {recall_custom}")
print(f"F1 Score: {f1_custom}")
print("\nScikit-learn AdaBoost Metrics:")
print(f"Precision: {precision_sklearn}")
print(f"Recall: {recall_sklearn}")
print(f"F1 Score: {f1_sklearn}")