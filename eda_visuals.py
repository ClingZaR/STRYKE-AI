import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("resampled_train_data.csv")

# Handle missing values
df["bmi"].fillna(df["bmi"].median(), inplace=True)
df["smoking_status"].fillna("Unknown", inplace=True)

# Target variable distribution
sns.countplot(data=df, x="stroke")
plt.title("Stroke Class Distribution")
plt.savefig("plots/stroke_distribution.png")
plt.clf()

# Numeric distributions
for col in ["age", "avg_glucose_level", "bmi"]:
    sns.histplot(data=df, x=col, hue="stroke", kde=True, bins=30)
    plt.title(f"{col.capitalize()} Distribution by Stroke")
    plt.savefig(f"plots/{col}_distribution.png")
    plt.clf()

# Correlation heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("plots/correlation_heatmap.png")
plt.clf()
