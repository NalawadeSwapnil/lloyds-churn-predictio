import pandas as pd
file = "Customer_Churn_Data_Large.xlsx"
demo  = pd.read_excel(file, sheet_name="Customer_Demographics")
churn = pd.read_excel(file, sheet_name="Churn_Status")

print("=== SIZE OF DATA ===")
print(f"Rows: {demo.shape[0]}")
print(f"Columns: {demo.shape[1]}")

print("\n=== COLUMN NAMES ===")
print(demo.columns.tolist())
print(churn.columns.tolist())

print("\n=== DATA TYPES ===")
print(demo.dtypes)
print(churn.dtypes)

print("\n===Missing Values===")
print(demo.isnull().sum())
print(churn.isnull().sum())

print("\n===BASIC STATISTICS===")
print(demo.describe())

print("\n===CHURN COUNTS===")
print(churn["ChurnStatus"].value_counts())