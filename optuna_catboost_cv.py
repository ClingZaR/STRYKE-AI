import joblib
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Load data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# Optuna objective with StratifiedKFold CV
def optuna_objective(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'depth': trial.suggest_int("depth", 4, 8),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 2.0, 10.0),
        'subsample': trial.suggest_float("subsample", 0.7, 1.0),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.7, 1.0),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 25, 40),
        'iterations': trial.suggest_int("iterations", 50, 100),
        'random_seed': 42,
        'verbose': 0,
        'loss_function': 'Logloss'
    }
    model = CatBoostClassifier(**params)
    f1 = cross_val_score(model, X_train, y_train, scoring='f1', cv=StratifiedKFold(5)).mean()
    return f1

study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=30)

print("\n✅ Best Params (CV Optimized):")
print(study.best_params)

# Save the best params
joblib.dump(study.best_params, "artifacts/catboost_optuna_best_params.pkl")
