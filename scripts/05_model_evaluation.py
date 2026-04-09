import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- ISA III FIX: Use ticket-level dataset (one row per ticket) ---
# PROBLEM: The email-level dataset has multiple rows per ticket with
# identical features, causing train-test leakage in cross-validation.
# SOLUTION: Use the aggregated ticket-level dataset instead.
df = pd.read_csv("data/processed/isa3_ticket_level.csv")

# =====================================================================
# THE "HONEST" FEATURE SET
# We explicitly removed is_assigned, votes, and watches because they 
# act as structural proxies (target leakage). We are now testing PURE
# socio-technical communication signals.
# =====================================================================
safe_features = [
    'email_count_per_ticket', 'subject_length',
    'avg_sentiment', 'sentiment_variance', 'sentiment_trend',
    'priority_numeric',
    'unique_senders',
    # --- ISA III FIX: description_length moved to Model B (JIRA structural signal) ---
]

X_model = df[safe_features]
y = df['is_stalled']

# Dynamically calculate the imbalance ratio for XGBoost (~4.3)
neg = (y == 0).sum()
pos = (y == 1).sum()
dynamic_scale_pos_weight = neg / pos if pos > 0 else 1

print("=" * 60)
print(f"ISA III: HONEST MODEL EVALUATION PIPELINE")
print(f"Training on {len(df)} Unique Tickets")
print(f"Features: {len(safe_features)} (Pure Communication/Sentiment Signals)")
print("=" * 60)

# Define Candidate Models
models = {
    "Logistic Regression (Linear Baseline)": LogisticRegression(class_weight='balanced', max_iter=2000),
    "Random Forest (ISA II Architecture)": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "XGBoost (Tuned Architecture)": XGBClassifier(scale_pos_weight=dynamic_scale_pos_weight, random_state=42, eval_metric='logloss')
}

# Rigorous 5-Fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['recall', 'precision', 'f1', 'roc_auc']

for name, model in models.items():
    cv_results = cross_validate(model, X_model, y, cv=cv, scoring=scoring_metrics)
    print(f"\n{name}:")
    print(f"  Mean Recall (Stalled): {cv_results['test_recall'].mean():.3f}  (± {cv_results['test_recall'].std() * 2:.3f})")
    print(f"  Mean Precision:        {cv_results['test_precision'].mean():.3f}")
    print(f"  Mean ROC-AUC:          {cv_results['test_roc_auc'].mean():.3f}")

print("\n" + "=" * 60)
print(X_model.columns.tolist())  # should show exactly 7 features