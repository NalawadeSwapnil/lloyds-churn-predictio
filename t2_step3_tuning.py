import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     GridSearchCV)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, recall_score,
                              f1_score, classification_report,
                              confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("cleaned_data.csv")
X  = df.drop(columns=["ChurnStatus"])
y  = df["ChurnStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance training data
smote = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("=== WHAT IS HYPERPARAMETER TUNING? ===")
print("""
Every algorithm has settings called hyperparameters
These are NOT learned from data - WE set them

Example for Gradient Boosting:
  n_estimators  = how many trees to build (100? 200? 300?)
  learning_rate = how fast it learns (0.01? 0.05? 0.1?)
  max_depth     = how deep each tree goes (2? 3? 5?)

GridSearchCV tries EVERY combination
and finds which settings work best!
""")

# ── Define parameter grid ─────────────────────────
param_grid = {
    'n_estimators' : [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth'    : [2, 3, 5],
}

total_combinations = 3 * 3 * 3
print(f"Testing {total_combinations} combinations x 5 folds")
print(f"= {total_combinations * 5} model fits total")
print("(This will take 2-5 minutes, please wait...)\n")

# ── Run GridSearchCV ──────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',    # optimise for AUC
    n_jobs=-1,            # use all CPU cores
    verbose=1             # show progress
)

grid_search.fit(X_train_bal, y_train_bal)

print("\n=== TUNING RESULTS ===")
print(f"Best parameters : {grid_search.best_params_}")
print(f"Best AUC score  : {grid_search.best_score_*100:.1f}%")

# ── Train final model with best params ───────────
best_model = grid_search.best_estimator_

print("\n=== FINAL MODEL PERFORMANCE (threshold=0.3) ===")
y_prob     = best_model.predict_proba(X_test)[:, 1]
y_pred_50  = (y_prob >= 0.5).astype(int)
y_pred_40  = (y_prob >= 0.4).astype(int)
y_pred_30  = (y_prob >= 0.3).astype(int)

for thresh, y_pred in [("0.5", y_pred_50),
                        ("0.4", y_pred_40),
                        ("0.3", y_pred_30)]:
    cm  = confusion_matrix(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nThreshold {thresh}:")
    print(f"  Recall  : {rec*100:.1f}%")
    print(f"  F1      : {f1*100:.1f}%")
    print(f"  ROC-AUC : {auc*100:.1f}%")
    print(f"  Churners caught: {cm[1][1]} / {cm[1][0]+cm[1][1]}")

# ── Use threshold 0.3 as our final choice ────────
print("\n=== FINAL CHOSEN MODEL — THRESHOLD 0.3 ===")
y_pred_final = y_pred_30
cm = confusion_matrix(y_test, y_pred_final)

print("\nConfusion Matrix:")
print(f"                Predicted Stayed  Predicted Churned")
print(f"Actual Stayed       {cm[0][0]}                {cm[0][1]}")
print(f"Actual Churned      {cm[1][0]}                {cm[1][1]}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_final,
      target_names=["Stayed","Churned"]))

# ── Plot: Before vs After Tuning ──────────────────
before_auc    = 53.2   # from step 1b
after_auc     = roc_auc_score(y_test, y_prob) * 100
before_recall = 49.0   # from step 1b threshold 0.3
after_recall  = recall_score(y_test, y_pred_30) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(["Before Tuning", "After Tuning"],
            [before_auc, after_auc],
            color=["#e74c3c","#2ecc71"], edgecolor="white")
axes[0].set_title("ROC-AUC: Before vs After Tuning")
axes[0].set_ylabel("ROC-AUC (%)")
axes[0].set_ylim(0, 100)
for i, v in enumerate([before_auc, after_auc]):
    axes[0].text(i, v+0.5, f"{v:.1f}%",
                 ha="center", fontweight="bold")

axes[1].bar(["Before Tuning","After Tuning"],
            [before_recall, after_recall],
            color=["#e74c3c","#2ecc71"], edgecolor="white")
axes[1].set_title("Recall: Before vs After Tuning")
axes[1].set_ylabel("Recall (%)")
axes[1].set_ylim(0, 100)
for i, v in enumerate([before_recall, after_recall]):
    axes[1].text(i, v+0.5, f"{v:.1f}%",
                 ha="center", fontweight="bold")

plt.suptitle("Impact of Hyperparameter Tuning",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("t2_chart2_tuning.png")
plt.show()
print("\nTuning chart saved!")

# ── Save best params for report ───────────────────
best_params = grid_search.best_params_
best_params["best_auc"]    = round(after_auc, 1)
best_params["best_recall"] = round(after_recall, 1)
pd.DataFrame([best_params]).to_csv("best_params.csv", index=False)
print("Best params saved!")