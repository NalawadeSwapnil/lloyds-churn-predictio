import pandas as pd
import matplotlib.pyplot as plt

# Load the already merged data — one line!
df = pd.read_csv("merged_data.csv")

print("Data loaded successfully!")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ── CHART 1: How many customers churned? ───────
plt.figure(figsize=(6, 4))
df["ChurnStatus"].value_counts().plot(
    kind="bar",
    color=["green", "red"],
    edgecolor="white"
)
plt.title("How Many Customers Churned?")
plt.xlabel("0 = Stayed,  1 = Churned")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart1_churn_count.png")
plt.show()
print("Chart 1 saved!")

# ── CHART 2: Age distribution by churn ─────────
plt.figure(figsize=(8, 4))
df[df["ChurnStatus"] == 0]["Age"].hist(
    bins=15, alpha=0.6, color="green", label="Stayed"
)
df[df["ChurnStatus"] == 1]["Age"].hist(
    bins=15, alpha=0.6, color="red", label="Churned"
)
plt.title("Age Distribution — Stayed vs Churned")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("chart2_age.png")
plt.show()
print("Chart 2 saved!")

# ── CHART 3: Churn rate by Income Level ────────
plt.figure(figsize=(6, 4))
income_churn = df.groupby("IncomeLevel")["ChurnStatus"].mean() * 100
income_churn = income_churn.reindex(["Low", "Medium", "High"])
income_churn.plot(
    kind="bar",
    color=["orange", "steelblue", "purple"],
    edgecolor="white"
)
plt.title("Churn Rate by Income Level")
plt.xlabel("Income Level")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart3_income.png")
plt.show()
print("Chart 3 saved!")

# ── CHART 4: Login Frequency vs Churn ──────────
plt.figure(figsize=(6, 4))
df.boxplot(
    column="LoginFrequency",
    by="ChurnStatus",
    patch_artist=True
)
plt.title("Login Frequency — Stayed vs Churned")
plt.suptitle("")
plt.xlabel("0 = Stayed,  1 = Churned")
plt.ylabel("Login Frequency")
plt.tight_layout()
plt.savefig("chart4_login.png")
plt.show()
print("Chart 4 saved!")
