import joblib
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ✅ Load data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

os.makedirs("tuning_plots", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
sns.set(style="whitegrid")

# 🔹 OPTUNA (with 5-fold CV and simplified search space)
def optuna_objective(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.15),
        'depth': trial.suggest_int("depth", 4, 6),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 3.0, 10.0),
        'subsample': trial.suggest_float("subsample", 0.6, 0.85),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.6, 0.85),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 35, 60),
        'iterations': trial.suggest_int("iterations", 40, 100),
        'random_seed': 42,
        'verbose': 0,
        'loss_function': 'Logloss',
        'class_weights': [1.0, 10.0]  # optional stroke bias
    }
    model = CatBoostClassifier(**params)
    f1 = cross_val_score(model, X_train, y_train, scoring='f1', cv=StratifiedKFold(5)).mean()
    return f1

optuna_study = optuna.create_study(direction="maximize")
optuna_study.optimize(optuna_objective, n_trials=30)

optuna_model = CatBoostClassifier(**optuna_study.best_params, verbose=0)
optuna_model.fit(X_train, y_train)
optuna_f1 = f1_score(y_test, optuna_model.predict(X_test))
joblib.dump(optuna_model, "artifacts/catboost_optuna_best_model.pkl")

# 🔹 SKOPT (Bayesian Search)
skopt_search = {
    'learning_rate': Real(0.01, 0.15),
    'depth': Integer(4, 6),
    'l2_leaf_reg': Real(3.0, 10.0),
    'subsample': Real(0.6, 0.85),
    'colsample_bylevel': Real(0.6, 0.85),
    'min_data_in_leaf': Integer(35, 60),
    'iterations': Integer(40, 100)
}
skopt_model = BayesSearchCV(
    CatBoostClassifier(verbose=0, random_seed=42, class_weights=[1.0, 10.0]),
    search_spaces=skopt_search,
    n_iter=15,
    scoring='f1',
    cv=StratifiedKFold(2),
    random_state=42,
    verbose=0
)
skopt_model.fit(X_train, y_train)
skopt_f1 = f1_score(y_test, skopt_model.best_estimator_.predict(X_test))

# 🔹 RandomizedSearchCV
random_grid = {
    'learning_rate': np.linspace(0.01, 0.15, 5),
    'depth': list(range(4, 7)),
    'l2_leaf_reg': np.linspace(3.0, 10.0, 5),
    'subsample': np.linspace(0.6, 0.85, 4),
    'colsample_bylevel': np.linspace(0.6, 0.85, 4),
    'min_data_in_leaf': list(range(35, 61, 5)),
    'iterations': list(range(40, 101, 10))
}
rand_model = RandomizedSearchCV(
    CatBoostClassifier(verbose=0, random_seed=42, class_weights=[1.0, 10.0]),
    param_distributions=random_grid,
    n_iter=15,
    scoring='f1',
    cv=StratifiedKFold(2),
    random_state=42,
    verbose=0
)
rand_model.fit(X_train, y_train)
rand_f1 = f1_score(y_test, rand_model.best_estimator_.predict(X_test))

# 🔹 Plotting
plt.figure(figsize=(8, 4))
sns.barplot(x=["Optuna", "skopt", "RandomizedSearch"], y=[optuna_f1, skopt_f1, rand_f1], palette="rocket")
plt.title("F1 Score Comparison - CatBoost Tuning (Regularized)")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
for i, f1 in enumerate([optuna_f1, skopt_f1, rand_f1]):
    plt.text(i, f1 + 0.01, f"{f1:.3f}", ha='center')
plt.tight_layout()
plt.savefig("tuning_plots/catboost_tuning_comparison.png")
plt.close()

# 🔹 Summary
print("📊 Plot saved: tuning_plots/catboost_tuning_comparison.png")
print("✅ Optuna Best Params:", optuna_study.best_params)
print("✅ skopt Best Params:", skopt_model.best_params_)
print("✅ RandomizedSearch Best Params:", rand_model.best_params_)
