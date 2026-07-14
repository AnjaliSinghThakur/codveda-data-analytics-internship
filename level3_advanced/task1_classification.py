"""
Codveda Data Analytics Internship - Level 3 (Advanced)
Task 1: Predictive Modeling (Classification)
Dataset: Churn Prediction Data (churn-bigml-80.csv = train, churn-bigml-20.csv = test)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TRAIN_PATH = "/home/claude/dataset1/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv"
TEST_PATH = "/home/claude/dataset1/Data Set For Task/Churn Prdiction Data/churn-bigml-20.csv"
PLOT_DIR = "/home/claude/work/plots"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print("Train shape:", train.shape, " Test shape:", test.shape)

def preprocess(df, encoders=None, fit=False):
    df = df.copy()
    cat_cols = ["State", "International plan", "Voice mail plan"]
    if fit:
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
    else:
        for c in cat_cols:
            df[c] = df[c].map(lambda v: v if v in encoders[c].classes_ else encoders[c].classes_[0])
            df[c] = encoders[c].transform(df[c])
    df["Churn"] = df["Churn"].astype(int)
    return df, encoders

# 1. Preprocess: handle categorical variables + feature scaling
train_p, encoders = preprocess(train, fit=True)
test_p, _ = preprocess(test, encoders=encoders, fit=False)

X_train = train_p.drop(columns=["Churn"])
y_train = train_p["Churn"]
X_test = test_p.drop(columns=["Churn"])
y_test = test_p["Churn"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train and test multiple classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
}

results = []
for name, m in models.items():
    if name == "Logistic Regression":
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_test_scaled)
    else:
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    })

results_df = pd.DataFrame(results)
print("\nModel comparison (default hyperparameters):\n", results_df)
results_df.to_csv("/home/claude/work/level3/classification_model_comparison.csv", index=False)

# 4. Hyperparameter tuning using grid search (Random Forest - best baseline typically)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest params (Random Forest, GridSearch):", grid.best_params_)
best_model = grid.best_estimator_
best_preds = best_model.predict(X_test)

tuned_metrics = {
    "accuracy": accuracy_score(y_test, best_preds),
    "precision": precision_score(y_test, best_preds),
    "recall": recall_score(y_test, best_preds),
    "f1": f1_score(y_test, best_preds),
}
print("Tuned Random Forest metrics:", tuned_metrics)

# Plots
# Confusion matrix for the tuned best model
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t1_confusion_matrix.png", dpi=120)
plt.close()

# Model comparison bar chart
plt.figure(figsize=(8, 5))
results_melted = results_df.melt(id_vars="model", var_name="metric", value_name="score")
sns.barplot(data=results_melted, x="metric", y="score", hue="model")
plt.title("Classification Model Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t1_model_comparison.png", dpi=120)
plt.close()

# Feature importance from tuned random forest
importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importances.head(10).plot(kind="barh", color="#2E86AB")
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t1_feature_importance.png", dpi=120)
plt.close()

print("\nClassification task complete. Plots saved to", PLOT_DIR)
