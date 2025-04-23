# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import joblib

# Load dataset
df = pd.read_csv("full_data.csv")
print("📥 Loaded data from full_data.csv")
print("Initial shape:", df.shape)

# Step 1: Fill missing values
df["bmi"].fillna(df["bmi"].median(), inplace=True)
df["smoking_status"].fillna("Unknown", inplace=True)
print("\n🧹 Filled missing values:")
print(" - BMI missing filled with median")
print(" - Smoking status missing filled with 'Unknown'")

# Step 2: Feature engineering (currently disabled)
# df["age_hypertension"] = df["age"] * df["hypertension"]
# print("🧠 Engineered feature 'age_hypertension'")

# Step 3: Encode categorical features
label_encoders = {}
categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
print("\n🔠 Encoding categorical features:")
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f" - Encoded '{col}' → classes: {list(le.classes_)}")

# Step 4: Normalize numerical features
scale_columns = ["age", "avg_glucose_level", "bmi"]
scaler = MinMaxScaler()
df[scale_columns] = scaler.fit_transform(df[scale_columns])
print("\n📏 Normalized numerical features using MinMaxScaler:")
for col in scale_columns:
    print(f" - {col} (range: {df[col].min():.2f} to {df[col].max():.2f})")

# Step 5: Save outputs
os.makedirs("artifacts", exist_ok=True)
df.to_csv("preprocessed_data.csv", index=False)
joblib.dump(label_encoders, "artifacts/label_encoders.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
print("\n✅ Data preprocessing complete.")
print(" - Saved to: preprocessed_data.csv")
print(" - Label encoders → artifacts/label_encoders.pkl")
print(" - Scaler → artifacts/scaler.pkl")
