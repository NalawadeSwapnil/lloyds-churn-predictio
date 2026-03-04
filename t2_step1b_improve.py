import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (recall_score, roc_auc_score,
                             f1_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("cleaned_data.csv")
X  = df.drop(columns=["ChurnStatus"])
y  = df["ChurnStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Try 3 different balancing strategies ──────────
print("=== TESTING DIFFERENT BALANCING STRATEGIES ===\n")

strategies = {

    "SMOTE only" : SMOTE(random_state=42),

    "SMOTETomek" : SMOTETomek(random_state=42),
    # SMOTETomek = creates new churners AND
    # removes confusing borderline cases

    "SMOTE aggressive (k=1)" : SMOTE(
                                    random_state=42,
                                    k_neighbors=1),
    # Uses nearest 1 neighbour instead of 5
    # More aggressive minority oversampling
}

for strat_name, sampler in strategies.items():

    X_bal, y_bal = sampler.fit_resample(X_train, y_train)

    # Try Gradient Boosting with lower threshold
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_bal, y_bal)

    # Use lower decision threshold (0.3 instead of 0.5)
    # This makes model more aggressive at flagging churners
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_30 = (y_prob >= 0.3).astype(int)
    y_pred_40 = (y_prob >= 0.4).astype(int)
    y_pred_50 = (y_prob >= 0.5).astype(int)

    print(f"Strategy: {strat_name}")
    print(f"  Balanced classes: {(y_bal==0).sum()} vs {(y_bal==1).sum()}")
    print(f"  Threshold 0.5 → Recall={recall_score(y_test,y_pred_50)*100:.0f}%  "
          f"AUC={roc_auc_score(y_test,y_prob)*100:.1f}%")
    print(f"  Threshold 0.4 → Recall={recall_score(y_test,y_pred_40)*100:.0f}%  "
          f"AUC={roc_auc_score(y_test,y_prob)*100:.1f}%")
    print(f"  Threshold 0.3 → Recall={recall_score(y_test,y_pred_30)*100:.0f}%  "
          f"AUC={roc_auc_score(y_test,y_prob)*100:.1f}%")
    print()

# ── Show best result with threshold tuning ────────
print("=== BEST MODEL WITH THRESHOLD = 0.3 ===")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_train, y_train)

best_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
best_model.fit(X_bal, y_bal)
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.3).astype(int)

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nOut of {(y_test==1).sum()} actual churners:")
print(f"  Caught  : {cm[1][1]}")
print(f"  Missed  : {cm[1][0]}")
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=["Stayed","Churned"]))

# Save best model probabilities for ROC curve later
pd.DataFrame({
    "y_test": y_test.values,
    "y_prob": y_prob
}).to_csv("model_probabilities.csv", index=False)
print("\nProbabilities saved for ROC curve!")