import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Define features and target
target_col = "stroke"
all_features = [col for col in df.columns if col != target_col]

X = df[all_features]
y = df[target_col]

# Train baseline model with all features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

baseline_acc = accuracy_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred)

print(f"\n✅ Baseline Accuracy: {baseline_acc:.4f}")
print(f"✅ Baseline F1 Score: {baseline_f1:.4f}\n")

# Feature importance ranking
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("📊 Feature Importance Ranking:")
print(importance_df.to_string(index=False))

# Compare performance after removing each feature
print("\n🔍 Evaluating feature drop impact:")
for feature in all_features:
    reduced_features = [f for f in all_features if f != feature]
    X_train_r, X_test_r = X_train[reduced_features], X_test[reduced_features]

    model_r = RandomForestClassifier(n_estimators=100, random_state=42)
    model_r.fit(X_train_r, y_train)
    y_pred_r = model_r.predict(X_test_r)

    acc = accuracy_score(y_test, y_pred_r)
    f1 = f1_score(y_test, y_pred_r)

    print(f"- Removed '{feature}': Accuracy={acc:.4f}, F1={f1:.4f}, ΔAcc={acc-baseline_acc:.4f}")
