import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ✅ Load SMOTE-split dataset
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# ✅ Tune CatBoost
model = CatBoostClassifier(verbose=0, random_state=42)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'iterations': [100, 200]
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# ✅ Evaluate best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Best Parameters:", grid.best_params_)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ F1 Score:", f1_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
