"""
Codveda Data Analytics Internship - Level 1 (Basic)
Task 1: Data Cleaning and Preprocessing
Dataset: Stock Prices Data Set.csv (raw dataset with missing values)
"""
import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/dataset1/Data Set For Task/2) Stock Prices Data Set.csv"
OUT_PATH = "/home/claude/work/level1/stock_prices_cleaned.csv"

# 1. Load the dataset using pandas
df = pd.read_csv(RAW_PATH)
print("Original shape:", df.shape)
print("\nData types:\n", df.dtypes)

# 2. Identify missing values
print("\nMissing values per column:\n", df.isnull().sum())
missing_pct = (df.isnull().sum() / len(df) * 100).round(3)
print("\nMissing value % per column:\n", missing_pct)

# 3. Handle missing values
# open/high/low have a small number of missing values (<15 rows out of ~497k).
# Strategy: fill with the previous trading day's value for the same stock symbol
# (forward-fill within each symbol group), which is standard practice for price series.
df = df.sort_values(["symbol", "date"])
for col in ["open", "high", "low"]:
    df[col] = df.groupby("symbol")[col].transform(lambda s: s.ffill().bfill())

print("\nMissing values after imputation:\n", df.isnull().sum())

# 4. Remove duplicate rows
dupes_before = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nDuplicate rows removed: {dupes_before}")

# 5. Standardize inconsistent data formats
# Convert date column to proper datetime type (was stored as text)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Standardize symbol column to uppercase, strip whitespace
df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

# Ensure numeric columns are proper numeric dtype
num_cols = ["open", "high", "low", "close", "volume"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sanity check: high should be >= low; flag/report any violations (data-quality check)
bad_rows = (df["high"] < df["low"]).sum()
print(f"\nRows where high < low (data quality flag): {bad_rows}")

# Final report
print("\nFinal cleaned shape:", df.shape)
print("Final missing values:", df.isnull().sum().sum())
print("Final duplicate rows:", df.duplicated().sum())

df.to_csv(OUT_PATH, index=False)
print(f"\nCleaned dataset saved to {OUT_PATH}")
