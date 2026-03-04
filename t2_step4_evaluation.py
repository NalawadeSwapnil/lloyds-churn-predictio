import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             precision_recall_curve)
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Load & prepare data ───────────────────────────
df = pd.read_csv("cleaned_data.csv")
X  = df.drop(columns=["ChurnStatus"])
y  = df["ChurnStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Load best params from tuning step
best = pd.read_csv("best_params.csv").iloc[0]

# ── Train final model ─────────────────────────────
model = GradientBoostingClassifier(
    n_estimators  = int(best["n_estimators"]),
    learning_rate = float(best["learning_rate"]),
    max_depth     = int(best["max_depth"]),
    random_state  = 42
)
model.fit(X_train_bal, y_train_bal)

y_prob       = model.predict_proba(X_test)[:, 1]
y_pred_final = (y_prob >= 0.3).astype(int)

# ─────────────────────────────────────────────────
# METRIC 1 — Confusion Matrix
# ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print("=== METRIC 1: CONFUSION MATRIX ===")
print(f"""
                  Predicted
                Stayed  Churned
Actual Stayed  [  {TN}   |   {FP}  ]
Actual Churned [  {FN}   |   {TP}   ]

TN={TN} Correctly told customer would STAY   ✅
TP={TP}  Correctly told customer would CHURN  ✅
FP={FP}  Wrongly said would churn (false alarm) ⚠️
FN={FN}  Missed a churner — most costly!        ❌
""")

# ─────────────────────────────────────────────────
# METRIC 2 — All Scores
# ─────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final, zero_division=0)
recall    = recall_score(y_test, y_pred_final)
f1        = f1_score(y_test, y_pred_final)
auc       = roc_auc_score(y_test, y_prob)

print("=== METRIC 2: ALL SCORES ===")
print(f"Accuracy  : {accuracy*100:.1f}%")
print(f"Precision : {precision*100:.1f}%")
print(f"Recall    : {recall*100:.1f}%")
print(f"F1 Score  : {f1*100:.1f}%")
print(f"ROC-AUC   : {auc*100:.1f}%")

print("""
=== WHAT EACH METRIC MEANS ===
Accuracy  = overall correct predictions
            (misleading with imbalanced data!)

Precision = of all predicted churners,
            how many actually churned?
            (are our alerts trustworthy?)

Recall    = of all actual churners,
            how many did we catch?
            (most important for the bank!)

F1 Score  = balance of precision & recall
            (useful single summary number)

ROC-AUC   = ability to rank churners above
            non-churners (0.5=random, 1=perfect)
""")

# ─────────────────────────────────────────────────
# CHART 1 — Confusion Matrix Visual
# ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(["Predicted\nStayed","Predicted\nChurned"], fontsize=11)
ax.set_yticklabels(["Actual\nStayed","Actual\nChurned"],  fontsize=11)
labels = [["TN","FP"],["FN","TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{cm[i][j]}",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i][j] > cm.max()/2 else "black")
ax.set_title("Confusion Matrix — Final Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("t2_chart3_confusion.png")
plt.show()
print("Confusion matrix chart saved!")

# ─────────────────────────────────────────────────
# CHART 2 — ROC Curve
# ─────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2980b9", linewidth=2.5,
        label=f"Model (AUC = {auc*100:.1f}%)")
ax.plot([0,1],[0,1], color="gray", linestyle="--",
        linewidth=1.5, label="Random guess (AUC = 50%)")

# Mark threshold 0.3 point
idx = np.argmin(np.abs(thresholds - 0.3))
ax.scatter(fpr[idx], tpr[idx], color="red", s=100, zorder=5,
           label=f"Threshold = 0.3")

ax.set_xlabel("False Positive Rate\n(% of non-churners wrongly flagged)")
ax.set_ylabel("True Positive Rate\n(% of churners correctly caught)")
ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig("t2_chart4_roc.png")
plt.show()
print("ROC curve saved!")

# ─────────────────────────────────────────────────
# CHART 3 — Precision Recall Curve
# ─────────────────────────────────────────────────
prec_curve, rec_curve, thresh_curve = precision_recall_curve(
    y_test, y_prob
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_curve, prec_curve, color="#27ae60", linewidth=2.5)
ax.axhline(y=y_test.mean(), color="gray", linestyle="--",
           label=f"Baseline ({y_test.mean()*100:.0f}% churn rate)")

# Mark threshold 0.3
idx2 = np.argmin(np.abs(thresh_curve - 0.3))
ax.scatter(rec_curve[idx2], prec_curve[idx2],
           color="red", s=100, zorder=5,
           label="Threshold = 0.3")

ax.set_xlabel("Recall (% of churners caught)")
ax.set_ylabel("Precision (% of alerts that are real churners)")
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig("t2_chart5_precision_recall.png")
plt.show()
print("Precision-Recall curve saved!")

# ─────────────────────────────────────────────────
# CHART 4 — Metrics Bar Chart
# ─────────────────────────────────────────────────
metrics = {
    "Accuracy" : accuracy*100,
    "Precision": precision*100,
    "Recall"   : recall*100,
    "F1 Score" : f1*100,
    "ROC-AUC"  : auc*100,
}

colors = ["#3498db","#e67e22","#e74c3c","#9b59b6","#1abc9c"]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics.keys(), metrics.values(),
              color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Metrics Summary",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
ax.axhline(y=50, color="gray", linestyle="--",
           alpha=0.5, label="50% baseline")
ax.legend()
plt.tight_layout()
plt.savefig("t2_chart6_metrics.png")
plt.show()
print("Metrics chart saved!")

# ─────────────────────────────────────────────────
# Save all metrics to CSV for report
# ─────────────────────────────────────────────────
pd.DataFrame([{
    "Accuracy"  : round(accuracy*100,1),
    "Precision" : round(precision*100,1),
    "Recall"    : round(recall*100,1),
    "F1_Score"  : round(f1*100,1),
    "ROC_AUC"   : round(auc*100,1),
    "TN": TN, "FP": FP, "FN": FN, "TP": TP
}]).to_csv("final_metrics.csv", index=False)
print("\nAll metrics saved to final_metrics.csv!")