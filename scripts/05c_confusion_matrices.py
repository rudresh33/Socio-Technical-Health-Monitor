"""
Generate confusion matrices for Model A and Model B using a single 80-20 train-test split.
This complements the cross-validation evaluation by showing model behaviour on a held-out sample.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    f1_score, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load ticket-level dataset
df = pd.read_csv("data/processed/final_ticket_level.csv")

# Define feature sets
MODEL_A_FEATURES = [
    'email_count_per_ticket', 'subject_length', 'avg_sentiment',
    'sentiment_variance', 'sentiment_trend', 'priority_numeric', 'unique_senders'
]

MODEL_B_FEATURES = MODEL_A_FEATURES + [
    'description_length',
    'type_Bug', 'type_Improvement', 'type_New Feature', 'type_Sub-task',
    'type_Task', 'type_Test', 'type_Wish',
    'proj_HADOOP', 'proj_HDFS', 'proj_MAPREDUCE', 'proj_YARN'
]

os.makedirs("charts", exist_ok=True)

def evaluate_with_confusion_matrix(features, model_name, output_path):
    """Train Model on 80% and evaluate confusion matrix on held-out 20%."""
    X = df[features].values
    y = df['is_stalled'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Test set size:        {len(y_test)} tickets")
    print(f"  Stalled in test:    {int(sum(y_test))}")
    print(f"  Resolved in test:   {int(len(y_test) - sum(y_test))}")
    print(f"Recall (Stalled):     {recall:.3f}")
    print(f"Precision (Stalled):  {precision:.3f}")
    print(f"F1-Score (Stalled):   {f1:.3f}")
    print(f"ROC-AUC:              {auc:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Resolved  Stalled")
    print(f"  Actual Resolved   {cm[0,0]:>8}  {cm[0,1]:>7}")
    print(f"  Actual Stalled    {cm[1,0]:>8}  {cm[1,1]:>7}")

    # Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Resolved/Active", "Stalled"],
        yticklabels=["Resolved/Active", "Stalled"],
        annot_kws={"size": 16, "weight": "bold"},
        cbar_kws={"shrink": 0.8}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(
        f"{model_name} — Confusion Matrix\n"
        f"(80-20 Stratified Train-Test Split, Test n={len(y_test)})",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output_path}")
    return cm, {"recall": recall, "precision": precision, "f1": f1, "auc": auc}


# Generate for both models
cm_a, metrics_a = evaluate_with_confusion_matrix(
    MODEL_A_FEATURES,
    "Model A (Communication-Only, 7 Features)",
    "charts/confusion_matrix_model_a_split.png"
)

cm_b, metrics_b = evaluate_with_confusion_matrix(
    MODEL_B_FEATURES,
    "Model B (Communication + Structural, 19 Features)",
    "charts/confusion_matrix_model_b_split.png"
)

print("\n" + "="*60)
print("SUMMARY (Single Train-Test Split)")
print("="*60)
print(f"{'Metric':<15} {'Model A':>12} {'Model B':>12} {'Diff':>10}")
print("-"*60)
for metric in ["recall", "precision", "f1", "auc"]:
    a, b = metrics_a[metric], metrics_b[metric]
    diff = b - a
    print(f"{metric:<15} {a:>12.3f} {b:>12.3f} {diff:>+10.3f}")
