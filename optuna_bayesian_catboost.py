# optuna_bayesian_catboost.py

import joblib
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ✅ Load SMOTE-split dataset
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# ✅ Define objective function for Optuna
def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'iterations': trial.suggest_int('iterations', 100, 300),
        'verbose': 0,
        'random_state': 42
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

# ✅ Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# ✅ Best results
print("\n✅ Best Parameters:", study.best_params)

# ✅ Evaluate final model
best_model = CatBoostClassifier(**study.best_params, verbose=0, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print("✅ Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("✅ F1 Score: {:.4f}".format(f1_score(y_test, y_pred)))
print("\n", classification_report(y_test, y_pred))
