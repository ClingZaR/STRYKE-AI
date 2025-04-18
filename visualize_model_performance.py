# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("preprocessed_data.csv")
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Train CatBoost
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_resampled, y_resampled)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.3f})".format(roc_auc_score(y_test, y_probs)))
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/roc_curve.png")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/precision_recall_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# Print quick stats
f1 = f1_score(y_test, y_pred)
print("✅ Visualization complete. F1 Score: {:.4f}".format(f1))
print("📊 Plots saved in /plots: roc_curve.png, precision_recall_curve.png, confusion_matrix.png")
