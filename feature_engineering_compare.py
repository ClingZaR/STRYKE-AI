import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")


# Create output directory
os.makedirs("plots", exist_ok=True)

# Features and target
target = "stroke"
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Baseline model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

baseline_acc = accuracy_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred)

print(f"\n✅ Baseline Accuracy: {baseline_acc:.4f}")
print(f"✅ Baseline F1 Score: {baseline_f1:.4f}")

# Feature importance
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n📊 Feature Importance Ranking:")
print(importance_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", dodge=False, legend=False, palette="viridis")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("plots/randomforest_feature_importance.png")
plt.close()

# Evaluate feature drop
drop_results = []
print("\n🔍 Evaluating feature drop impact:")
for feature in features:
    reduced = [f for f in features if f != feature]
    model_r = RandomForestClassifier(n_estimators=100, random_state=42)
    model_r.fit(X_train[reduced], y_train)
    y_pred_r = model_r.predict(X_test[reduced])
    acc = accuracy_score(y_test, y_pred_r)
    f1 = f1_score(y_test, y_pred_r)
    delta_acc = acc - baseline_acc
    drop_results.append((feature, acc, f1, delta_acc))
    print(f"- Removed '{feature}': Accuracy={acc:.4f}, F1={f1:.4f}, ΔAcc={delta_acc:+.4f}")

# Plot drop impact
drop_df = pd.DataFrame(drop_results, columns=["feature", "acc", "f1", "delta_acc"]).sort_values(by="delta_acc")

plt.figure(figsize=(10, 6))
sns.barplot(data=drop_df, x="delta_acc", y="feature", hue="feature", dodge=False, legend=False, palette="coolwarm")
plt.title("Accuracy Impact of Removing Each Feature")
plt.xlabel("Δ Accuracy")
plt.tight_layout()
plt.savefig("plots/randomforest_feature_removal_impact.png")
plt.close()
