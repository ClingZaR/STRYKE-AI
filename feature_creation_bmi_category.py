import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Step 1: Create BMI category
def categorize_bmi(bmi):
    if bmi < 0.185:
        return 'Underweight'
    elif bmi < 0.25:
        return 'Normal'
    elif bmi < 0.30:
        return 'Overweight'
    else:
        return 'Obese'

df["bmi_category"] = df["bmi"].apply(categorize_bmi)

# Step 2: Encode category
le = LabelEncoder()
df["bmi_category_encoded"] = le.fit_transform(df["bmi_category"])

# Prepare all 3 versions
features_A = [col for col in df.columns if col != "stroke" and col != "bmi_category" and col != "bmi_category_encoded"]
features_B = [col for col in df.columns if col != "stroke" and col != "bmi" and col != "bmi_category"]
features_C = [col for col in df.columns if col != "stroke" and col != "bmi_category"]

def evaluate(features, label):
    X = df[features]
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"📦 {label}: Accuracy={acc:.4f}, F1={f1:.4f}")

print("\n✅ Comparing BMI Feature Versions")
evaluate(features_A, "A: Using BMI only")
evaluate(features_B, "B: Using BMI Category only")
evaluate(features_C, "C: Using both BMI + BMI Category")
