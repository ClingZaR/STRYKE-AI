import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Load dataset
df = pd.read_csv("full_data.csv")

# Step 1: Fill missing values
df["bmi"].fillna(df["bmi"].median(), inplace=True)
df["smoking_status"].fillna("Unknown", inplace=True)

# Step 2: Encode categorical features
label_encoders = {}
categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Normalize numerical features
scale_columns = ["age", "avg_glucose_level", "bmi"]
scaler = MinMaxScaler()
df[scale_columns] = scaler.fit_transform(df[scale_columns])

# Save preprocessed version
df.to_csv("preprocessed_data.csv", index=False)

# Save encoders & scaler for inference
import joblib
os.makedirs("artifacts", exist_ok=True)
joblib.dump(label_encoders, "artifacts/label_encoders.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("✅ Data preprocessing complete. Saved to preprocessed_data.csv and artifacts/")
