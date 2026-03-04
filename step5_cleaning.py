import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load our saved merged data
df = pd.read_csv("merged_data.csv")

print("=== ORIGINAL DATA ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.head())

# ─────────────────────────────────────────
# STEP 5a — Find & Fix Outliers
# ─────────────────────────────────────────
# An outlier is an extreme value that could
# mess up our machine learning model
# We use the IQR method to find them

print("\n=== STEP 5a: OUTLIER DETECTION ===")

# We only check numerical columns
num_cols = ["Age", "TotalSpent", "NumTransactions",
            "AvgSpend", "NumInteractions",
            "NumComplaints", "LoginFrequency"]

for col in num_cols:
    Q1  = df[col].quantile(0.25)  # bottom 25%
    Q3  = df[col].quantile(0.75)  # top 25%
    IQR = Q3 - Q1                 # the middle range

    lower = Q1 - 1.5 * IQR       # anything below this = outlier
    upper = Q3 + 1.5 * IQR       # anything above this = outlier

    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"{col}: {outliers} outliers found → capping them")

    # Cap the outliers (bring them inside the boundary)
    df[col] = df[col].clip(lower=lower, upper=upper)

print("\nOutliers fixed! ✅")

# ─────────────────────────────────────────
# STEP 5b — Encode Text Columns
# ─────────────────────────────────────────
# Machine learning cannot read words like
# "Male" or "Low" — we convert them to numbers

print("\n=== STEP 5b: ENCODING TEXT COLUMNS ===")

# Drop columns we don't need for prediction
df = df.drop(columns=["CustomerID", "LastLoginDate"])

# One-hot encoding turns one column into multiple
# Example:
# Gender     →  Gender_M   Gender_F
# M          →     1          0
# F          →     0          1

df = pd.get_dummies(df, columns=["Gender",
                                  "MaritalStatus",
                                  "IncomeLevel",
                                  "ServiceUsage"])

print("Text columns encoded! ✅")
print(f"Columns now: {df.shape[1]}")
print(df.columns.tolist())

# ─────────────────────────────────────────
# STEP 5c — Normalise Numerical Columns
# ─────────────────────────────────────────
# Age goes from 18-69
# TotalSpent goes from 0-5000
# These different scales confuse ML models
# Normalising brings everything to the same scale

print("\n=== STEP 5c: NORMALISING ===")

# Separate our target column before scaling
target = df["ChurnStatus"]
df     = df.drop(columns=["ChurnStatus"])

# These are the columns we will normalise
cols_to_scale = ["Age", "TotalSpent", "NumTransactions",
                 "AvgSpend", "NumInteractions",
                 "NumComplaints", "LoginFrequency"]

# StandardScaler makes mean=0, std=1 for each column
scaler         = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Put the target column back
df["ChurnStatus"] = target

print("Normalisation done! ✅")
print("\n=== FINAL CLEANED DATASET ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.head())

# Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)
print("\n✅ Cleaned data saved as cleaned_data.csv!")