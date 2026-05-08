import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- MODEL B — Communication + Structural Features ---
# PURPOSE: Compare against Model A (pure communication signals) to test
# whether knowing WHICH project or issue type a ticket belongs to
# improves stall prediction beyond communication patterns alone.

df = pd.read_csv("data/processed/final_ticket_level.csv")

# Model A features (pure email communication signals)
communication_features = [
    'email_count_per_ticket', 'subject_length',
    'avg_sentiment', 'sentiment_variance', 'sentiment_trend',
    'priority_numeric',
    'unique_senders',
]

# Structural features (JIRA metadata — project, issuetype, description)
structural_features = ['description_length'] + [col for col in df.columns if col.startswith('type_') or col.startswith('proj_')]

# Model B = communication + structural
all_features = communication_features + structural_features

X_model = df[all_features]
y = df['is_stalled']

neg = (y == 0).sum()
pos = (y == 1).sum()
dynamic_scale_pos_weight = neg / pos if pos > 0 else 1

print("=" * 60)
print(f"MODEL B — COMMUNICATION + STRUCTURAL FEATURES")
print(f"Training on {len(df)} Unique Tickets")
print(f"Communication features: {len(communication_features)}")
print(f"Structural features: {len(structural_features)} {structural_features}")
print(f"Total features: {len(all_features)}")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=dynamic_scale_pos_weight, random_state=42, eval_metric='logloss'),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['recall', 'precision', 'f1', 'roc_auc']

for name, model in models.items():
    cv_results = cross_validate(model, X_model, y, cv=cv, scoring=scoring_metrics)
    print(f"\n{name}:")
    print(f"  Mean Recall (Stalled): {cv_results['test_recall'].mean():.3f}  (+/- {cv_results['test_recall'].std() * 2:.3f})")
    print(f"  Mean Precision:        {cv_results['test_precision'].mean():.3f}")
    print(f"  Mean F1:               {cv_results['test_f1'].mean():.3f}")
    print(f"  Mean ROC-AUC:          {cv_results['test_roc_auc'].mean():.3f}")

print("\n" + "=" * 60)
print("COMPARISON GUIDE:")
print("  Model A (05_model_evaluation.py): 7 communication features only")
print("  Model B (this file): 7 communication + 12 structural features")
print("  If B >> A: structural factors dominate prediction")
print("  If B ~= A: communication signals are sufficient")
print("=" * 60)


# --- Aggregated 5-fold confusion matrix for Model B ---
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)

all_y_true = []
all_y_pred = []
for train_idx, test_idx in cv.split(X_model, y):
    clf = LogisticRegression(class_weight='balanced', max_iter=2000)
    clf.fit(X_model.iloc[train_idx], y.iloc[train_idx])
    all_y_true.extend(y.iloc[test_idx])
    all_y_pred.extend(clf.predict(X_model.iloc[test_idx]))

cm = confusion_matrix(all_y_true, all_y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Resolved/Active", "Stalled"],
            yticklabels=["Resolved/Active", "Stalled"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Model B — Confusion Matrix (Aggregated 5-Fold)")
plt.tight_layout()
plt.savefig("charts/confusion_matrix_model_b.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrix saved to charts/confusion_matrix_model_b.png")
