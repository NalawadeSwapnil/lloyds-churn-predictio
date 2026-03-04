import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

file = "Customer_Churn_Data_Large.xlsx"

# Load and merge all sheets
demo   = pd.read_excel(file, sheet_name="Customer_Demographics")
txn    = pd.read_excel(file, sheet_name="Transaction_History")
svc    = pd.read_excel(file, sheet_name="Customer_Service")
online = pd.read_excel(file, sheet_name="Online_Activity")
churn  = pd.read_excel(file, sheet_name="Churn_Status")

txn_summary = txn.groupby("CustomerID").agg(
    TotalSpent      = ("AmountSpent", "sum"),
    NumTransactions = ("TransactionID", "count"),
    AvgSpend        = ("AmountSpent", "mean")
).reset_index()

svc_summary = svc.groupby("CustomerID").agg(
    NumInteractions = ("InteractionID", "count"),
    NumComplaints   = ("InteractionType",
                       lambda x: (x == "Complaint").sum())
).reset_index()

df = demo.merge(txn_summary, on="CustomerID", how="left")
df = df.merge(svc_summary,   on="CustomerID", how="left")
df = df.merge(online,        on="CustomerID", how="left")
df = df.merge(churn,         on="CustomerID", how="left")

# Drop columns not needed
df = df.drop(columns=["CustomerID", "LastLoginDate"])

# Encode text columns
df = pd.get_dummies(df, columns=["Gender",
                                  "MaritalStatus",
                                  "IncomeLevel",
                                  "ServiceUsage"])

# ─────────────────────────────────────
# Helper function to test each method
# ─────────────────────────────────────
def test_model(df_input, method_name):

    X = df_input.drop(columns=["ChurnStatus"])
    y = df_input["ChurnStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(
        X_train, y_train
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    TP  = cm[1][1]  # churned correctly caught
    FN  = cm[1][0]  # churned missed

    print(f"\n{'━'*40}")
    print(f"Method : {method_name}")
    print(f"{'━'*40}")
    print(f"Accuracy          : {acc*100:.1f}%")
    print(f"Churners caught   : {TP} out of {TP+FN}")
    print(f"Churners missed   : {FN}")
    print(f"Confusion Matrix  :")
    print(cm)

# ─────────────────────────────────────
# METHOD 1 — Fill with ZERO
# ─────────────────────────────────────
df_zero = df.copy()
df_zero["NumInteractions"] = df_zero["NumInteractions"].fillna(0)
df_zero["NumComplaints"]   = df_zero["NumComplaints"].fillna(0)
test_model(df_zero, "Fill with ZERO")

# ─────────────────────────────────────
# METHOD 2 — Fill with MEAN
# ─────────────────────────────────────
df_mean = df.copy()
df_mean["NumInteractions"] = df_mean["NumInteractions"].fillna(
    df_mean["NumInteractions"].mean()
)
df_mean["NumComplaints"] = df_mean["NumComplaints"].fillna(
    df_mean["NumComplaints"].mean()
)
print(f"\nMean NumInteractions : {df['NumInteractions'].mean():.2f}")
print(f"Mean NumComplaints   : {df['NumComplaints'].mean():.2f}")
test_model(df_mean, "Fill with MEAN")

# ─────────────────────────────────────
# METHOD 3 — Fill with MEDIAN
# ─────────────────────────────────────
df_median = df.copy()
df_median["NumInteractions"] = df_median["NumInteractions"].fillna(
    df_median["NumInteractions"].median()
)
df_median["NumComplaints"] = df_median["NumComplaints"].fillna(
    df_median["NumComplaints"].median()
)
print(f"\nMedian NumInteractions : {df['NumInteractions'].median():.2f}")
print(f"Median NumComplaints   : {df['NumComplaints'].median():.2f}")
test_model(df_median, "Fill with MEDIAN")

print(f"\n{'━'*40}")
print("COMPARISON COMPLETE! ✅")
print(f"{'━'*40}")