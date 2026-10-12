"""
Codveda Data Analytics Internship - Level 2 (Intermediate)
Task 1: Regression Analysis
Dataset: House Prediction Data Set.csv (Boston Housing) - predict MEDV (house price)
          from RM (average number of rooms per dwelling)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

RAW_PATH = "/home/claude/dataset1/Data Set For Task/4) house Prediction Data Set.csv"
PLOT_DIR = "/home/claude/work/plots"
COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
           "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df = pd.read_csv(RAW_PATH, header=None, sep=r"\s+", names=COLUMNS)

# Predict house price (MEDV) from number of rooms (RM) -- simple linear regression
X = df[["RM"]]
y = df["MEDV"]

# 1. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

# 2. Fit a linear regression model using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Interpret coefficients and evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nIntercept: {model.intercept_:.4f}")
print(f"Coefficient (RM): {model.coef_[0]:.4f}")
print(f"Interpretation: Each additional room is associated with an average "
      f"increase of ${model.coef_[0]*1000:.0f} in median house value.")
print(f"\nR-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Also fit a multiple regression model using all features for comparison
X_multi = df.drop(columns=["MEDV"])
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(Xm_train, ym_train)
ym_pred = model_multi.predict(Xm_test)
r2_multi = r2_score(ym_test, ym_pred)
rmse_multi = np.sqrt(mean_squared_error(ym_test, ym_pred))
print(f"\n[Comparison] Multiple regression (all 13 features): R2 = {r2_multi:.4f}, RMSE = {rmse_multi:.4f}")

# Plot: regression line over test data
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, alpha=0.6, label="Actual", color="#2E86AB")
order = np.argsort(X_test["RM"].values)
plt.plot(X_test["RM"].values[order], y_pred[order], color="#F18F01", linewidth=2, label="Predicted (fit line)")
plt.xlabel("Average Rooms per Dwelling (RM)")
plt.ylabel("Median House Value (MEDV, $1000s)")
plt.title(f"Simple Linear Regression: RM vs MEDV (R2={r2:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l2_t1_regression_fit.png", dpi=120)
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color="#A23B72")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted MEDV")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l2_t1_residuals.png", dpi=120)
plt.close()

print("\nRegression analysis complete. Plots saved to", PLOT_DIR)
