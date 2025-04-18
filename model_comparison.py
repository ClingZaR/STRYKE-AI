# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Load preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    
    acc = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    f1 = cross_val_score(pipeline, X, y, cv=kfold, scoring=make_scorer(f1_score))
    
    print("\n🔎 {}:".format(name))
    print("  Accuracy: {:.4f} ± {:.4f}".format(acc.mean(), acc.std()))
    print("  F1 Score: {:.4f} ± {:.4f}".format(f1.mean(), f1.std()))
