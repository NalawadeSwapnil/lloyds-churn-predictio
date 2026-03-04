import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Load cleaned data from Task 1 ─────────────────
df = pd.read_csv("cleaned_data.csv")

X = df.drop(columns=["ChurnStatus"])
y = df["ChurnStatus"]

# ── Train / Test split ────────────────────────────
# stratify=y ensures both splits have same churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # ← keeps churn ratio balanced
)

# ── Apply SMOTE to training data only ────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Training rows after SMOTE : {X_train_bal.shape[0]}")
print(f"Testing rows              : {X_test.shape[0]}")

# ── Define 4 algorithms to compare ───────────────
algorithms = {
    "Logistic Regression" : LogisticRegression(
                                max_iter=1000,
                                random_state=42,
                                class_weight="balanced"),

    "Decision Tree"       : DecisionTreeClassifier(
                                random_state=42,
                                class_weight="balanced"),

    "Random Forest"       : RandomForestClassifier(
                                n_estimators=100,
                                random_state=42),

    "Gradient Boosting"   : GradientBoostingClassifier(
                                n_estimators=100,
                                random_state=42),
}

# ── Train & evaluate each algorithm ──────────────
print("\n=== ALGORITHM COMPARISON ===")
print(f"{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-" * 72)

results = {}

for name, model in algorithms.items():

    # Train the model
    model.fit(X_train_bal, y_train_bal)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy" : round(acc*100, 1),
        "Precision": round(prec*100, 1),
        "Recall"   : round(rec*100, 1),
        "F1"       : round(f1*100, 1),
        "ROC-AUC"  : round(auc*100, 1),
    }

    print(f"{name:<22} {acc*100:>8.1f}% {prec*100:>9.1f}% "
          f"{rec*100:>7.1f}% {f1*100:>7.1f}% {auc*100:>8.1f}%")

# ── Print winner ──────────────────────────────────
print("\n=== BEST ALGORITHM BY ROC-AUC ===")
best = max(results, key=lambda x: results[x]["ROC-AUC"])
print(f"Winner: {best} with ROC-AUC = {results[best]['ROC-AUC']}%")

print("\n=== BEST ALGORITHM BY RECALL ===")
best_recall = max(results, key=lambda x: results[x]["Recall"])
print(f"Winner: {best_recall} with Recall = {results[best_recall]['Recall']}%")

print("""
=== WHY RECALL MATTERS MOST ===
In churn prediction for a bank:
- Missing a churner (False Negative) = COSTLY
  Bank loses the customer forever
- Wrong churn alert (False Positive) = CHEAP
  Bank sends a retention offer unnecessarily

So we prioritise RECALL over raw Accuracy!
""")