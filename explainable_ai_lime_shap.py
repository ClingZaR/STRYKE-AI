# explainable_ai_lime_shap.py (for CatBoost)

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from catboost import CatBoostClassifier
from lime import lime_tabular
import shap

# Create directory for visualizations
os.makedirs("explainability_plots", exist_ok=True)

# ✅ Load data
print("🔄 Loading data...")
X_train, X_test, y_train, y_test = joblib.load("train_test_smote.pkl")
feature_names = X_train.columns.tolist()

# ✅ Load pretrained final model
print("📦 Loading pretrained model from joblib...")
model = joblib.load("artifacts/final_catboost_model.pkl")


# ✅ Make predictions for reference
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.50).astype(int)  # Using 0.50 threshold as in your script

# ======= LIME EXPLANATIONS =======
print("\n✅ Generating LIME explanations...")

# Initialize LIME explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=["No Event", "Event"],
    mode='classification',
    random_state=42
)

# Generate explanations for a few test instances
num_samples_to_explain = 3
for i in range(num_samples_to_explain):
    lime_instance = lime_explainer.explain_instance(
        data_row=X_test.iloc[i].values,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    # Save LIME visualization to HTML
    lime_instance.save_to_file(f"explainability_plots/lime_explanation_{i}.html")
    
    # Show LIME explanation details in console output
    print(f"\n📊 LIME explanation for test sample #{i}:")
    print(f"   True label: {y_test.iloc[i]}, Predicted: {y_pred[i]} (Prob: {y_probs[i]:.4f})")
    for feat, val in lime_instance.as_list():
        print(f"   {feat}: {val:.4f}")
        
    # Create visualization for this instance
    plt.figure(figsize=(10, 6))
    lime_instance.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f"explainability_plots/lime_plot_{i}.png")
    plt.close()

# ======= SHAP EXPLANATIONS =======
print("\n✅ Generating SHAP explanations...")

# Create a subset of the training data to speed up SHAP calculations
shap_train_sample = X_train.sample(min(1000, len(X_train)), random_state=42)

# Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

# ✅ Global feature importance
print("   Calculating global SHAP values...")
shap_values = explainer.shap_values(X_test)

# ✅ SHAP Summary Plot (bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Global Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig("explainability_plots/shap_summary_bar.png")
plt.close()

# ✅ SHAP Summary Plot (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('Feature Impact on Predictions (SHAP)')
plt.tight_layout()
plt.savefig("explainability_plots/shap_summary_beeswarm.png")
plt.close()

# ✅ SHAP Individual Explanations
# Explain the same samples as we did with LIME for comparison
for i in range(num_samples_to_explain):
    plt.figure(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[i, :],
        X_test.iloc[i, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - Sample #{i}')
    plt.tight_layout()
    plt.savefig(f"explainability_plots/shap_force_plot_{i}.png")
    plt.close()

    # Generate SHAP decision plot for this instance
    plt.figure(figsize=(12, 6))
    shap.decision_plot(
        explainer.expected_value,
        shap_values[i, :],
        X_test.iloc[i, :],
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Decision Plot - Sample #{i}')
    plt.tight_layout()
    plt.savefig(f"explainability_plots/shap_decision_plot_{i}.png")
    plt.close()

# ✅ SHAP Dependence Plots for top features
# Identify top features based on mean absolute SHAP values
mean_shap_values = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(-mean_shap_values)[:5]  # Get top 5 features
top_features = [feature_names[i] for i in top_indices]

print(f"\n🔝 Top 5 features by SHAP importance:")
for i, feature in enumerate(top_features):
    print(f"   {i+1}. {feature} (importance: {mean_shap_values[top_indices[i]]:.4f})")
    
    # Create dependence plot for each top feature
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_indices[i],
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {feature}')
    plt.tight_layout()
    plt.savefig(f"explainability_plots/shap_dependence_{feature.replace(' ', '_')}.png")
    plt.close()

# ✅ Function to explain a new data point
def explain_prediction(model, sample, feature_names, lime_explainer, shap_explainer):
    """
    Explain a prediction for a new data point using both LIME and SHAP.
    """
    # Convert sample to 2D array if needed
    if isinstance(sample, pd.Series):
        sample_2d = sample.values.reshape(1, -1)
    else:
        sample_2d = np.array(sample).reshape(1, -1)

    # Get prediction and probabilities
    pred_proba = model.predict_proba(sample_2d)[0]
    prediction = "Stroke" if pred_proba[1] >= 0.3 else "No Stroke"
    
    print("\n🔍 Prediction Analysis:")
    print(f"Prediction: {prediction}")
    print(f"No Stroke Probability: {pred_proba[0]:.4f}")
    print(f"Stroke Probability: {pred_proba[1]:.4f}\n")

    # LIME Explanation
    print("🎯 LIME Explanation:")
    lime_exp = lime_explainer.explain_instance(
        sample.values if isinstance(sample, pd.Series) else sample,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    # Print LIME features
    for feat, val in lime_exp.as_list():
        print(f"{feat}: {val:.4f}")

    # SHAP Explanation
    print("\n🎯 SHAP Values:")
    shap_values = shap_explainer(sample_2d)[0]
    for fname, sv in zip(feature_names, shap_values.values):
        print(f"{fname}: {sv:.4f}")

    return {
        'prediction': prediction,
        'probabilities': pred_proba,
        'lime_explanation': lime_exp,
        'shap_values': shap_values
    }

# Example usage of the explain_prediction function
print("\n📊 Example of prediction explanation for a new sample:")
example_sample = X_test.iloc[0]  # Using first test sample as an example
explanation = explain_prediction(
    model=model,
    sample=example_sample,
    feature_names=feature_names,
    lime_explainer=lime_explainer,
    shap_explainer=explainer
)

print(f"   Predicted: {explanation['prediction']} with probability {explanation['probabilities'][1]:.4f}")
print("   Top LIME features:")
for feat, val in explanation['lime_explanation'].as_list()[:5]:
    print(f"     {feat}: {val:.4f}")
print("   Top SHAP features:")
for fname, sv in zip(feature_names, explanation['shap_values'].values[:5]):
    print(f"     {fname}: {sv:.4f}")

print("\n✅ All explanations and visualizations have been saved to 'explainability_plots/' directory")