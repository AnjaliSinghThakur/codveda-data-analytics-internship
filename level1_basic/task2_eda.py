"""
Codveda Data Analytics Internship - Level 1 (Basic)
Task 2: Exploratory Data Analysis (EDA)
Dataset: House Prediction Data Set.csv (Boston Housing dataset)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = "/home/claude/dataset1/Data Set For Task/4) house Prediction Data Set.csv"
PLOT_DIR = "/home/claude/work/plots"

COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
           "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df = pd.read_csv(RAW_PATH, header=None, sep=r"\s+", names=COLUMNS)
print("Shape:", df.shape)
print(df.head())

# 1. Summary statistics (mean, median, mode, std)
summary = pd.DataFrame({
    "mean": df.mean(),
    "median": df.median(),
    "mode": df.mode().iloc[0],
    "std": df.std(),
    "min": df.min(),
    "max": df.max(),
})
print("\nSummary statistics:\n", summary)
summary.to_csv("/home/claude/work/level1/eda_summary_statistics.csv")

# 2. Visualize distributions - histograms of key variables
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
key_cols = ["MEDV", "RM", "LSTAT", "CRIM", "AGE", "TAX"]
for ax, col in zip(axes.flat, key_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l1_t2_histograms.png", dpi=120)
plt.close()

# Boxplots to inspect outliers
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, ["MEDV", "CRIM", "LSTAT"]):
    sns.boxplot(y=df[col], ax=ax, color="#F18F01")
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l1_t2_boxplots.png", dpi=120)
plt.close()

# Scatter plot: RM (rooms) vs MEDV (price) - expect positive relation
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="RM", y="MEDV", color="#A23B72")
plt.title("Rooms per Dwelling vs Median House Value")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l1_t2_scatter_rm_medv.png", dpi=120)
plt.close()

# 3. Correlation between numerical features
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - House Features")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l1_t2_correlation_heatmap.png", dpi=120)
plt.close()

top_corr_with_target = corr["MEDV"].sort_values(ascending=False)
print("\nCorrelation with target (MEDV):\n", top_corr_with_target)
top_corr_with_target.to_csv("/home/claude/work/level1/eda_correlation_with_medv.csv")

print("\nEDA complete. Plots saved to", PLOT_DIR)
