# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Update this map according to your label encoding
gender_map = {0: "Female", 1: "Male", 2: "Other"}

# Evaluate model for each gender
for gender_value in df["gender"].unique():
    gender_name = gender_map.get(gender_value, "Unknown ({})".format(gender_value))
    subset = df[df["gender"] == gender_value]

    X = subset.drop(columns=["stroke", "gender"])
    y = subset["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nGender: {}".format(gender_name))
    print("  Accuracy: {:.4f}".format(acc))
    print("  F1 Score: {:.4f}".format(f1))
    print("  Sample Size: {}".format(len(subset)))
