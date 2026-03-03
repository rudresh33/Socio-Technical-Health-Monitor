# Socio-Technical Health Monitor

> Predicting developer wellbeing and task stalling using JIRA ticket metadata and email sentiment analysis.

**MScIDS Semester VI Capstone Project**  
Goa Business School, Goa University  
Guide: Dr. Swapnil Fadte

---

## Team

| Name | Roll No. |
|------|----------|
| Rudresh Achari | 2330 |
| Unnat Umarye | 2303 |
| Sarvadhnya Patil | 2321 |
| Samuel Bhandari | 2308 |
| Harsh Palyekar | 2329 |

---

## Overview

This project builds a machine learning system that monitors the socio-technical health of software development teams by analysing patterns in JIRA issue tracker data and developer email communications. The system predicts two outcomes:

1. **Developer sentiment** - inferring emotional state from the tone of email communications linked to JIRA tickets.
2. **Task stalling risk** - classifying whether an open ticket is likely to remain stalled based on sentiment features and ticket metadata.

The motivation is to provide engineering managers with early, interpretable signals of team stress and delivery risk, enabling proactive intervention before problems escalate.

---

## Dataset

- **Source:** Apache Software Foundation public JIRA archive + mailing list email threads
- **Projects covered:** Hadoop, HDFS, MapReduce
- **Time period:** January 2023 to December 2024
- **Records (post-preprocessing):** 7,051 tickets
- **Class distribution:** 6,860 Resolved (97.3%) / 191 Stalled (2.7%) - ratio ~35.9:1

### Variables

| Type | Variables |
|------|-----------|
| Structured | Priority (Blocker/Critical/Major/Minor/Trivial), Status, Days to Resolve, Email Volume |
| Unstructured | Email subject lines and message bodies |
| Derived | Behavior Score (Sentiment), Sentiment Variance, Sentiment Trend |

---

## Methodology

### Sentiment Extraction

Email text is processed using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, selected for its suitability with short-form professional communication. Each email produces a compound sentiment score in the range [-1.0, +1.0].

### Feature Engineering

Six features are used for classification:

| Feature | Description | Importance |
|---------|-------------|------------|
| `email_volume_per_ticket` | Min-max scaled count of emails per ticket | 0.2960 |
| `sentiment_trend` | 3-month rolling average of monthly sentiment scores | 0.2549 |
| `sentiment_variance` | Standard deviation of scores across all emails per ticket | 0.1883 |
| `behavior_score` | VADER compound score, normalised [-1.0, +1.0] | 0.1320 |
| `subject_length` | Normalised character count of email subject line | 0.0833 |
| `priority_numeric` | Ordinal encoding: Blocker=5 down to Trivial=1 | 0.0456 |

Communication-derived features (`email_volume_per_ticket`, `sentiment_trend`, `sentiment_variance`) account for **73.9% of total feature importance**, confirming that the NLP pipeline contributes substantially more predictive signal than structured JIRA metadata alone.

### Model

**Primary model:** Random Forest Classifier with recall-optimised threshold

| Parameter | Value |
|-----------|-------|
| Estimators | 200 |
| Max depth | 10 |
| Class weight | balanced |
| Train/test split | 80/20 stratified |
| Decision threshold | 0.35 (recall-optimised) |

### Results

Evaluated on a held-out test set of 1,411 records:

| Metric | Resolved (0) | Stalled (1) | Macro Avg |
|--------|-------------|-------------|-----------|
| Precision | 0.99 | 0.17 | 0.58 |
| Recall | 0.89 | **0.82** | 0.85 |
| F1-Score | 0.94 | 0.28 | 0.61 |

**Confusion Matrix:**

|  | Predicted Resolved | Predicted Stalled |
|--|-------------------|------------------|
| **True Resolved** | 1224 (TN) | 149 (FP) |
| **True Stalled** | 7 (FN) | 31 (TP) |

The model correctly identifies **31 of 38 stalled tickets** while missing only 7. A threshold of 0.35 (vs. the standard 0.50) is applied to maximise recall for the minority class, accepting a higher false positive rate as a deliberate trade-off - missing a genuine stall carries a higher operational cost than a false flag.

---

## Key Findings from EDA

- Developer sentiment remained broadly positive (mean = 0.28) across the 24-month observation period.
- Sentiment dropped sharply to **0.10** during the Hadoop 3.4.1 release candidate crunch (November 2024), recovering to **0.42** post-release.
- Critical tickets carry the **lowest median sentiment (0.23)**; Blocker tickets show an elevated score (0.38) attributed to relief following rapid resolution.
- Stalled tasks exhibit a **bimodal sentiment distribution** - high upper quartile (frustration release) and extended negative tail (sustained stress).
- Email volume is right-skewed: median 2 messages per ticket; high-volume tickets indicate contested discussions and are the **strongest stall predictor**.
- Priority designation is a **weak predictor** (importance 0.0456), suggesting teams do not consistently escalate priority in response to emerging delays.

---

## Project Structure

```
Socio-Technical-Health-Monitor/
|
|-- data/
|   |-- raw/                        # Raw JIRA and email archive data
|   +-- processed/                  # Cleaned and feature-engineered datasets
|
|-- scripts/
|   +-- random_forest_model.py      # Model training and evaluation script
|
|-- visuals/
|   |-- 1_sentiment_by_priority.png
|   |-- 2_stalled_vs_active.png
|   |-- 3_correlation_heatmap.png
|   |-- 5_monthly_sentiment_trend.png
|   |-- 6_email_volume_distribution.png
|   |-- 7_task_status_by_priority.png
|   +-- 8_confusion_matrix.png
|
|-- reports/                        # ISA II academic report and presentation
|
|-- requirements.txt
+-- README.md
```

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/rudresh33/Socio-Technical-Health-Monitor.git
cd Socio-Technical-Health-Monitor

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas
numpy
scikit-learn
vaderSentiment
matplotlib
seaborn
jupyter
```

---

## Usage

```bash
# Run the full Random Forest model pipeline
python3 scripts/random_forest_model.py
```

This will:
1. Load the enriched dataset
2. Print class distribution
3. Train the model with an 80/20 stratified split
4. Evaluate at threshold 0.35
5. Print the classification report, confusion matrix values, and feature importances
6. Save the confusion matrix plot to `visuals/8_confusion_matrix.png`

---

## Roadmap (ISA III)

- [ ] Hyperparameter optimisation via 5-fold stratified cross-validation
- [ ] XGBoost comparison model
- [ ] SHAP (SHapley Additive exPlanations) interpretability analysis
- [ ] Engineering manager dashboard prototype for stall risk communication
- [ ] Final dissertation report

---

## Academic Context

This repository is the codebase for a Semester VI MScIDS capstone project submitted under the ISA (Internal Semester Assessment) framework at Goa Business School, Goa University. The project is evaluated across three ISA milestones covering problem definition, data analysis and modelling, and final delivery.

---

## License

This project is submitted for academic assessment. All rights reserved by the authors. Please contact the team before reusing any part of this work.