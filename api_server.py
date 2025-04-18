# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from catboost import CatBoostClassifier
from lime import lime_tabular
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# ✅ FastAPI init
app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Static index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# ✅ Input schema
class InputData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

# ✅ Load data & model
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")
feature_columns = X_train.columns.tolist()

model = CatBoostClassifier(
    depth=6,
    learning_rate=0.016768689687909344,
    l2_leaf_reg=7,
    iterations=108,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train)

# ✅ LIME explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_columns,
    class_names=["No Stroke", "Stroke"],
    mode='classification'
)

# ✅ Categorical mappings
# ✅ Load label encoders
label_encoders = joblib.load("artifacts/label_encoders.pkl")


# ✅ Correct scaler: use real data, not SMOTE-resampled
scale_columns = ["age", "avg_glucose_level", "bmi"]
raw_df = pd.read_csv("full_data.csv")  # Must be the unscaled version
raw_df = raw_df.copy()
raw_df["bmi"].fillna(raw_df["bmi"].median(), inplace=True)
raw_df["smoking_status"].fillna("Unknown", inplace=True)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


scaler = MinMaxScaler()
scaler.fit(raw_df[scale_columns])

# ✅ Input preprocessing
def preprocess_input(data: dict):
    processed = data.copy()
    for col in label_encoders:
        processed[col] = label_encoders[col].transform([processed[col]])[0]

    df_input = pd.DataFrame([processed], columns=feature_columns)
    df_input[scale_columns] = scaler.transform(df_input[scale_columns])

    # Restore original BMI to categorize
    original_bmi = data["bmi"]
    bmi_category = get_bmi_category(original_bmi)

    return df_input, bmi_category


# ✅ Prediction endpoint
@app.post("/predict")
def predict_risk(input_data: InputData):
    input_df, bmi_category = preprocess_input(input_data.dict())
    y_prob = float(model.predict_proba(input_df)[0][1])
    y_pred = int(y_prob >= 0.40)

    exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0],
        predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_columns))
    )
    explanation = {k: round(v, 4) for k, v in exp.as_list()}

    return {
        "risk_score": round(y_prob, 4),
        "prediction": y_pred,
        "model": "catboost-optuna",
        "bmi_category": bmi_category,
        "lime_explanation": explanation
    }
