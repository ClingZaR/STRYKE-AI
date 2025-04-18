# threshold_tuning.py

import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# ✅ Load tuned model & test set
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# ✅ Best tuned CatBoost model from previous step
model = CatBoostClassifier(
    depth=8,
    iterations=200,
    l2_leaf_reg=1,
    learning_rate=0.1,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train)

# ✅ Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# ✅ Try different thresholds
thresholds = [0.5, 0.4, 0.35, 0.3, 0.25]
print("\n🎯 Threshold Tuning Results:\n")
for threshold in thresholds:
    y_pred_thresh = (y_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    print("Threshold: {:.2f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(threshold, f1, precision, recall))
