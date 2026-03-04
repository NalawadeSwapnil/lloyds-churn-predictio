import pandas as pd
# We import pandas and give it a nickname "pd"
# pandas is like Excel but in Python
import pandas as pd

# Tell Python where your Excel file is
# Since it's in the same folder, just write the filename
file = "Customer_Churn_Data_Large.xlsx"

# Load each sheet into its own variable
demo   = pd.read_excel(file, sheet_name="Customer_Demographics")
txn    = pd.read_excel(file, sheet_name="Transaction_History")
svc    = pd.read_excel(file, sheet_name="Customer_Service")
online = pd.read_excel(file, sheet_name="Online_Activity")
churn  = pd.read_excel(file, sheet_name="Churn_Status")

# Print what's inside each sheet
print("=== Demographics ===")
print(demo.head())

print("=== Transactions ===")
print(txn.head())

print("=== Churn Status ===")
print(churn.head())
