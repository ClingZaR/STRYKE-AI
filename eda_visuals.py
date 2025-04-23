# eda_visuals.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df_raw = pd.read_csv("full_data.csv")
df = df_raw.copy()  # Use raw data for more meaningful visuals

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

sns.set(style="whitegrid", palette="muted")

# -------------------- Bonus Visuals: Health Condition Trends --------------------

# 1. Hypertension vs Age
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df[df["hypertension"] == 1], x="age", label="Hypertension", fill=True, common_norm=False)
sns.kdeplot(data=df[df["hypertension"] == 0], x="age", label="No Hypertension", fill=True, common_norm=False)
plt.title("Age Distribution by Hypertension Status")
plt.legend()
plt.tight_layout()
plt.savefig("plots/age_vs_hypertension.png")
plt.clf()

# 2. Heart Disease vs Age
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df[df["heart_disease"] == 1], x="age", label="Heart Disease", fill=True, common_norm=False)
sns.kdeplot(data=df[df["heart_disease"] == 0], x="age", label="No Heart Disease", fill=True, common_norm=False)
plt.title("Age Distribution by Heart Disease")
plt.legend()
plt.tight_layout()
plt.savefig("plots/age_vs_heart_disease.png")
plt.clf()

# 3. BMI vs Age (Colored by Stroke)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="bmi", hue="stroke", alpha=0.6)
plt.title("BMI vs Age Colored by Stroke")
plt.tight_layout()
plt.savefig("plots/bmi_vs_age_by_stroke.png")
plt.clf()

# -------------------- Basic Distributions --------------------
# Stroke distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="stroke", data=df)
plt.title("Stroke Class Distribution")
plt.tight_layout()
plt.savefig("plots/stroke_distribution.png")
plt.clf()

# Numeric distributions by stroke
for col in ["age", "avg_glucose_level", "bmi"]:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue="stroke", bins=30, kde=True, stat="density", common_norm=False)
    plt.title(f"{col.capitalize()} Distribution by Stroke")
    plt.tight_layout()
    plt.savefig(f"plots/{col}_stroke_distribution.png")
    plt.clf()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.clf()

# -------------------- Categorical Stroke Rates --------------------
def stroke_rate_barplot(column, title, filename):
    stroke_rate_df = df.groupby(column)["stroke"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=stroke_rate_df, x=column, y="stroke")
    plt.title(title)
    plt.ylabel("Stroke Rate")
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.clf()

stroke_rate_barplot("gender", "Stroke Rate by Gender", "stroke_rate_by_gender")
stroke_rate_barplot("Residence_type", "Stroke Rate by Residence Type", "stroke_rate_by_residence")
stroke_rate_barplot("work_type", "Stroke Rate by Work Type", "stroke_rate_by_work")
stroke_rate_barplot("smoking_status", "Stroke Rate by Smoking Status", "stroke_rate_by_smoking")
stroke_rate_barplot("ever_married", "Stroke Rate by Marital Status", "stroke_rate_by_married")
stroke_rate_barplot("heart_disease", "Stroke Rate by Heart Disease", "stroke_rate_by_heart")
stroke_rate_barplot("hypertension", "Stroke Rate by Hypertension", "stroke_rate_by_hypertension")

# -------------------- Feature Relations --------------------
# Boxplot of glucose by stroke
plt.figure(figsize=(8, 5))
sns.boxplot(x="stroke", y="avg_glucose_level", data=df)
plt.title("Glucose Level by Stroke")
plt.tight_layout()
plt.savefig("plots/glucose_boxplot.png")
plt.clf()

# Boxplot of BMI by stroke
plt.figure(figsize=(8, 5))
sns.boxplot(x="stroke", y="bmi", data=df)
plt.title("BMI by Stroke")
plt.tight_layout()
plt.savefig("plots/bmi_boxplot.png")
plt.clf()

# Boxplot of age by stroke
plt.figure(figsize=(8, 5))
sns.boxplot(x="stroke", y="age", data=df)
plt.title("Age by Stroke")
plt.tight_layout()
plt.savefig("plots/age_boxplot.png")
plt.clf()

# -------------------- Advanced Analysis --------------------
# Stroke rate by age group using raw age
df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 70, 80, 120])
stroke_by_age_group = df.groupby("age_group")["stroke"].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="age_group", y="stroke", data=stroke_by_age_group)
plt.title("Stroke Rate by Age Group")
plt.xticks(rotation=45)
plt.ylabel("Stroke Rate")
plt.tight_layout()
plt.savefig("plots/stroke_by_age_group.png")
plt.clf()

# Pairplot with raw features
pairplot_features = ["age", "avg_glucose_level", "bmi", "stroke"]
sns.pairplot(df[pairplot_features], hue="stroke", diag_kind="kde", corner=True)
plt.savefig("plots/pairplot_core_features.png")
plt.clf()

print("✅ All EDA plots saved to /plots/")
