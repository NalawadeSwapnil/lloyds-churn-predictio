import pandas as pd

file = "Customer_Churn_Data_Large.xlsx"

# ── Load all 5 sheets ──────────────────────────
demo   = pd.read_excel(file, sheet_name="Customer_Demographics")
txn    = pd.read_excel(file, sheet_name="Transaction_History")
svc    = pd.read_excel(file, sheet_name="Customer_Service")
online = pd.read_excel(file, sheet_name="Online_Activity")
churn  = pd.read_excel(file, sheet_name="Churn_Status")

# ── Transaction sheet has MULTIPLE rows per customer ──
# We need to SUMMARISE it into ONE row per customer
# This is called "aggregation"
txn_summary = txn.groupby("CustomerID").agg(
    TotalSpent        = ("AmountSpent", "sum"),   # total money spent
    NumTransactions   = ("TransactionID", "count"), # how many transactions
    AvgSpend          = ("AmountSpent", "mean")   # average spend per transaction
).reset_index()

print("=== Transaction Summary ===")
print(txn_summary.head())

# ── Service sheet also has MULTIPLE rows per customer ──
svc_summary = svc.groupby("CustomerID").agg(
    NumInteractions = ("InteractionID", "count"),  # total contacts
    NumComplaints   = ("InteractionType", lambda x: (x == "Complaint").sum())
).reset_index()

print("\n=== Service Summary ===")
print(svc_summary.head())

# ── Now merge everything together one by one ──
# Start with demographics as the base
df = demo.merge(txn_summary, on="CustomerID", how="left")
df = df.merge(svc_summary,   on="CustomerID", how="left")
df = df.merge(online,        on="CustomerID", how="left")
df = df.merge(churn,         on="CustomerID", how="left")

print("\n=== FINAL MERGED DATASET ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.columns.tolist())
print(df.head())

# Fill missing service data with 0
# (no interaction = 0 complaints, not missing)
df["NumInteractions"] = df["NumInteractions"].fillna(0)
df["NumComplaints"]   = df["NumComplaints"].fillna(0)

# Check missing values across all columns
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Save the merged dataset as a CSV file
df.to_csv("merged_data.csv", index=False)
print("Merged data saved as merged_data.csv!")