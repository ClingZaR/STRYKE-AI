import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier

# ✅ Load data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# ✅ Optuna Best Params (CV Tuned)
params = {'learning_rate': 0.14537841552636244, 'depth': 5, 'l2_leaf_reg': 3.0185695134279777, 'subsample': 0.741662230555452, 'colsample_bylevel': 0.7854167674188678, 'min_data_in_leaf': 56, 'iterations': 100}
print(y_test.value_counts())

# ✅ Train model
model = CatBoostClassifier(**params)
model.fit(X_train, y_train)

# ✅ Cross-validation to detect overfitting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()

# ✅ Evaluate training set
y_train_pred = model.predict(X_train)
train_f1 = f1_score(y_train, y_train_pred)
train_acc = accuracy_score(y_train, y_train_pred)

# ✅ Evaluate test set
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_thresh = (y_probs >= 0.50).astype(int)
test_f1 = f1_score(y_test, y_pred_thresh)
test_acc = accuracy_score(y_test, y_pred_thresh)
test_precision = precision_score(y_test, y_pred_thresh)
test_recall = recall_score(y_test, y_pred_thresh)

# ✅ Final report
print(f"\n🎯 FINAL TESTING (Optuna-Tuned CatBoost + Threshold 0.50)")
print(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
print(f"Test Accuracy : {test_acc:.4f} | F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")
print(f"CV F1 (5-fold): {cv_f1:.4f}")

# ✅ Overfitting Check
print("\n📊 Overfitting Check:")
print(f"Train vs Test F1 Gap : {train_f1 - test_f1:.4f}")
print(f"Train vs CV F1 Gap   : {train_f1 - cv_f1:.4f}")
print(f"CV vs Test F1 Gap    : {cv_f1 - test_f1:.4f}")

if train_f1 - test_f1 > 0.15 or train_f1 - cv_f1 > 0.15:
    print("⚠️  Overfitting Detected — consider more regularization or simpler model.")
else:
    print("✅ No significant overfitting detected.")

# ✅ Classification Report
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred_thresh))

# ✅ Visuals
os.makedirs("finaltesting_plots", exist_ok=True)
sns.set(style="whitegrid")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.figure()
plt.plot(recall, precision, lw=2, color='purple')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("finaltesting_plots/precision_recall_curve.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("finaltesting_plots/roc_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_thresh)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("finaltesting_plots/confusion_matrix.png")
plt.close()

print("📊 Visuals saved to: finaltesting_plots/")
joblib.dump(model, "artifacts/final_catboost_model.pkl")
print("✅ Model saved to: artifacts/final_catboost_model.pkl")
