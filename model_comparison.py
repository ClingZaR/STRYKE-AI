# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict

# Setup
os.makedirs("model_plots", exist_ok=True)
sns.set(style="whitegrid")

# Load data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")

# Define base models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
    "Linear SVM": LinearSVC(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Define ensemble models
try:
    ensemble_voting = VotingClassifier(estimators=[
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"]),
        ('log', models["Logistic Regression"])
    ], voting='soft')

    ensemble_stacking = StackingClassifier(estimators=[
        ('svc', models["SVM (Linear)"] if "SVM (Linear)" in models else SVC(kernel='linear')),
        ('gb', models["Gradient Boosting"]),
    ], final_estimator=LogisticRegression())

    models["Voting Classifier"] = ensemble_voting
    models["Stacking Classifier"] = ensemble_stacking
except Exception as e:
    print("Error setting up ensemble models:", e)

# Evaluate models
results = []
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append((name, acc, f1))
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Summary table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"]).dropna()

# Save barplot for F1 Score
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.sort_values("F1 Score", ascending=False), x="F1 Score", y="Model", palette="crest")
plt.title("Model Comparison by F1 Score")
plt.tight_layout()
plt.savefig("model_plots/model_comparison_f1_score.png")
plt.close()

# Save barplot for Accuracy
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.sort_values("Accuracy", ascending=False), x="Accuracy", y="Model", palette="mako")
plt.title("Model Comparison by Accuracy")
plt.tight_layout()
plt.savefig("model_plots/model_comparison_accuracy.png")
plt.close()

# Print summary
print("\n📊 Model Evaluation Summary:")
print(results_df.to_string(index=False))
