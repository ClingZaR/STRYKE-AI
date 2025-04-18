# explainable_ai_lime_shap.py

import joblib
import numpy as np
import shap
from lime import lime_tabular
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")
feature_names = X_train.columns

# Load Optuna-tuned CatBoost model
model = CatBoostClassifier(
    depth=6,
    learning_rate=0.016768689687909344,
    l2_leaf_reg=7,
    iterations=108,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train)

# ✅ LIME Explanation
lime_exp = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=["No Stroke", "Stroke"],
    mode='classification'
)

i = 0  # Test case index
lime_instance = lime_exp.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=model.predict_proba
)

print("✅ LIME explanation for test sample #{}:".format(i))
for feat, val in lime_instance.as_list():
    print(f"{feat}: {val:.4f}")

lime_instance.save_to_file("lime_explanation_{}.html".format(i))

# ✅ SHAP Explanation (global)
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)

# Plot summary
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_summary_bar.png")
plt.close()

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_dot.png")
plt.close()

print("\n✅ SHAP global summary plots saved: shap_summary_bar.png & shap_summary_dot.png")
