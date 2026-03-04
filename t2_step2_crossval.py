import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     cross_val_score)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("cleaned_data.csv")
X  = df.drop(columns=["ChurnStatus"])
y  = df["ChurnStatus"]

print("=== WHAT IS CROSS VALIDATION? ===")
print("""
Normal train/test split:
  [------ Train 80% ------][-- Test 20% --]
  Only tested on ONE set of 200 rows

Cross Validation (5-fold):
  Fold 1: [Test][Train][Train][Train][Train]
  Fold 2: [Train][Test][Train][Train][Train]
  Fold 3: [Train][Train][Test][Train][Train]
  Fold 4: [Train][Train][Train][Test][Train]
  Fold 5: [Train][Train][Train][Train][Test]

Model is tested on EVERY row exactly once
Gives us a much more reliable performance estimate!
""")

# ── Build a Pipeline ──────────────────────────────
# A Pipeline chains steps together so SMOTE
# is applied correctly inside each fold
# (prevents data leakage!)

pipeline = Pipeline([
    ('smote', SMOTETomek(random_state=42)),
    ('model', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# ── 5-Fold Stratified Cross Validation ───────────
# Stratified = keeps churn ratio in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== RUNNING 5-FOLD CROSS VALIDATION ===")
print("(This may take 30-60 seconds...)\n")

# Test with ROC-AUC score
auc_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

# Test with Recall score
recall_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='recall',
    n_jobs=-1
)

# Test with F1 score
f1_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

print("=== CROSS VALIDATION RESULTS ===")
print(f"\nROC-AUC per fold : {[round(s*100,1) for s in auc_scores]}")
print(f"Mean ROC-AUC     : {auc_scores.mean()*100:.1f}%")
print(f"Std Dev          : +/- {auc_scores.std()*100:.1f}%")

print(f"\nRecall per fold  : {[round(s*100,1) for s in recall_scores]}")
print(f"Mean Recall      : {recall_scores.mean()*100:.1f}%")
print(f"Std Dev          : +/- {recall_scores.std()*100:.1f}%")

print(f"\nF1 per fold      : {[round(s*100,1) for s in f1_scores]}")
print(f"Mean F1          : {f1_scores.mean()*100:.1f}%")
print(f"Std Dev          : +/- {f1_scores.std()*100:.1f}%")

print("""
=== HOW TO READ THESE RESULTS ===
Small std dev = model is CONSISTENT across different data ✅
Large std dev = model is UNSTABLE, results vary a lot     ❌
""")

# ── Plot Cross Validation Results ────────────────
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f"Fold {i+1}" for i in range(5)]
x = range(len(folds))

ax.plot(x, auc_scores*100,  'bo-', label='ROC-AUC',  linewidth=2, markersize=8)
ax.plot(x, recall_scores*100,'rs-', label='Recall',   linewidth=2, markersize=8)
ax.plot(x, f1_scores*100,   'g^-', label='F1 Score', linewidth=2, markersize=8)

ax.axhline(y=auc_scores.mean()*100,    color='blue',  linestyle='--', alpha=0.4)
ax.axhline(y=recall_scores.mean()*100, color='red',   linestyle='--', alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylabel("Score (%)")
ax.set_title("5-Fold Cross Validation Results")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("t2_chart1_crossval.png")
plt.show()
print("Cross validation chart saved!")

# Save scores for report
pd.DataFrame({
    'Fold'   : folds,
    'ROC_AUC': (auc_scores*100).round(1),
    'Recall' : (recall_scores*100).round(1),
    'F1'     : (f1_scores*100).round(1)
}).to_csv("crossval_scores.csv", index=False)
print("Scores saved!")