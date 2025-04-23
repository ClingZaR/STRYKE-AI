# cross_validation_compare.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

# Define models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# CV setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1 = make_scorer(f1_score)

# Evaluate
print("🔁 Cross-Validation Results (with SMOTE in pipeline):\n")
for name, model in models.items():
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    acc = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    f1s = cross_val_score(pipeline, X, y, cv=kfold, scoring=f1)

    print(f"📦 {name}")
    print("  Accuracy: {:.4f} ± {:.4f}".format(acc.mean(), acc.std()))
    print("  F1 Score: {:.4f} ± {:.4f}".format(f1s.mean(), f1s.std()))
    print()
