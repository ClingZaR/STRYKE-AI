# threshold_tuning.py

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os

# ✅ Load data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# ✅ Thresholds to evaluate
thresholds = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]

# ✅ Output dir
os.makedirs("threshold_plots", exist_ok=True)
sns.set(style="whitegrid")

# ✅ Train CatBoost model
params = {
    'learning_rate': 0.14993852352262632,
    'depth': 6,
    'l2_leaf_reg': 3.970288761243971,
    'subsample': 0.6071043367692867,
    'colsample_bylevel': 0.7614836624101209,
    'min_data_in_leaf': 40,
    'iterations': 95,
    'random_seed': 42,
    'verbose': 0,
    'loss_function': 'Logloss',
    'class_weights': [1.0, 10.0]  # Optional stroke boosting
}

model = CatBoostClassifier(**params)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
# ✅ Evaluate thresholds
results = []
print("\n🎯 CatBoost Threshold Tuning:\n")
for threshold in thresholds:
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    print(f"Threshold: {threshold:.2f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    results.append((threshold, f1, precision, recall))

# ✅ Create DataFrame
df = pd.DataFrame(results, columns=["Threshold", "F1 Score", "Precision", "Recall"])

# ✅ Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="Threshold", y="F1 Score", marker='o', label="F1", color="orange")
sns.lineplot(data=df, x="Threshold", y="Precision", marker='o', label="Precision", linestyle='--')
sns.lineplot(data=df, x="Threshold", y="Recall", marker='o', label="Recall", linestyle='--')
plt.title("CatBoost - Threshold Tuning")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("threshold_plots/catboost_threshold_tuning.png")
plt.close()

print("\n✅ Plot saved to threshold_plots/catboost_threshold_tuning.png")
