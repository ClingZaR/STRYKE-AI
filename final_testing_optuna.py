# final_testing_optuna.py

import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# ✅ Load SMOTE-split data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# ✅ Best tuned parameters from Optuna
params = {
    'depth': 6,
    'learning_rate': 0.016768689687909344,
    'l2_leaf_reg': 7,
    'iterations': 108,
    'verbose': 0,
    'random_state': 42
}

# ✅ Train model
model = CatBoostClassifier(**params)
model.fit(X_train, y_train)

# ✅ Evaluate on training set (overfitting check)
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# ✅ Predict probabilities on test
y_probs = model.predict_proba(X_test)[:, 1]

# ✅ Apply threshold = 0.40
y_pred_thresh = (y_probs >= 0.40).astype(int)

# ✅ Final evaluation
test_acc = accuracy_score(y_test, y_pred_thresh)
test_f1 = f1_score(y_test, y_pred_thresh)
test_precision = precision_score(y_test, y_pred_thresh)
test_recall = recall_score(y_test, y_pred_thresh)

# ✅ Report
print("🎯 FINAL TESTING (Optuna-Tuned CatBoost + Threshold 0.40)")
print("Train Accuracy: {:.4f} | F1: {:.4f}".format(train_acc, train_f1))
print("Test Accuracy : {:.4f} | F1: {:.4f}".format(test_acc, test_f1))
print("Test Precision: {:.4f} | Recall: {:.4f}".format(test_precision, test_recall))
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred_thresh))
