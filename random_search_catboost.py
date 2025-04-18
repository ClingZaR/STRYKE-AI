# random_search_catboost.py

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from catboost import CatBoostClassifier

# Load SMOTE-split dataset
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# Define CatBoost model
model = CatBoostClassifier(verbose=0, random_state=42)

# Define hyperparameter space (wide and randomized)
param_dist = {
    'depth': np.arange(4, 10),
    'learning_rate': np.linspace(0.01, 0.2, 20),
    'iterations': [100, 150, 200, 300],
    'l2_leaf_reg': np.arange(1, 10)
}

# Randomized Search
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 random combos
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit
random_search.fit(X_train, y_train)

# Evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n✅ Best Random Search Parameters:", random_search.best_params_)
print("✅ Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("✅ F1 Score: {:.4f}".format(f1_score(y_test, y_pred)))
print("\n", classification_report(y_test, y_pred))
