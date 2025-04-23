# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("preprocessed_data.csv")



# Drop target column
columns_to_drop = ["stroke"]

# Show original class balance
print("\nOriginal Class Distribution:")
print(df["stroke"].value_counts(normalize=True))

# Separate features and target
X = df.drop(columns=columns_to_drop)
y = df["stroke"]

print("Features used for SMOTE training:\n", X.columns.tolist())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Save resampled dataset
resampled_export = pd.DataFrame(X_train_resampled, columns=X.columns)
resampled_export["stroke"] = y_train_resampled.values
resampled_export.to_csv("resampled_train_data.csv", index=False)
print("\n💾 Resampled training data saved to resampled_train_data.csv")

# Show new class balance
print("\nResampled Class Distribution:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# Train model for evaluation
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save split
joblib.dump((X_train_resampled, X_test, y_train_resampled, y_test), "train_test_smote.pkl")
print("\n💾 Saved SMOTE train/test split to train_test_smote.pkl")

print("\n✅ Evaluation After SMOTE:")
print("Accuracy: {:.4f}".format(acc))
print("F1 Score: {:.4f}".format(f1))
