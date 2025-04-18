# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Split for 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ✅ Simple 80/20 with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)

acc_80_20 = accuracy_score(y_test, y_pred)
f1_80_20 = f1_score(y_test, y_pred)

# ✅ 5-Fold Cross Validation with SMOTE in pipeline
pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_cv = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
f1_cv = cross_val_score(pipeline, X, y, cv=kfold, scoring=make_scorer(f1_score))

# ✅ Results
print("\n🧪 80/20 Split:")
print("Accuracy: {:.4f}".format(acc_80_20))
print("F1 Score: {:.4f}".format(f1_80_20))

print("\n🔁 5-Fold Cross-Validation:")
print("Accuracy: {:.4f} ± {:.4f}".format(acc_cv.mean(), acc_cv.std()))
print("F1 Score: {:.4f} ± {:.4f}".format(f1_cv.mean(), f1_cv.std()))
