import pandas as pd
import numpy as np

# --- CONFIGURATION ---
input_file = "data/master_project_dataset.csv"
output_file = "data/enriched_project_dataset.csv"

print("Loading Master Dataset...")
df = pd.read_csv(input_file)

print("Starting Feature Engineering...")

# 1. Convert strings to datetime objects safely with UTC normalization
df['created'] = pd.to_datetime(df['created'], errors='coerce', utc=True)
df['resolutiondate'] = pd.to_datetime(df['resolutiondate'], errors='coerce', utc=True)
df['email_date'] = pd.to_datetime(df['email_date'], errors='coerce', utc=True)

# Drop rows where we cannot determine when the email or ticket was created
df = df.dropna(subset=['created', 'email_date'])

print(f"  Rows after date cleaning: {len(df)}")

# 2. Feature: days_to_resolve
# How long did the ticket take to close? NaN if still open.
df['days_to_resolve'] = (df['resolutiondate'] - df['created']).dt.days

# 3. Feature: is_stalled (Refined)
# A ticket is stalled if: it is neither Resolved nor Closed,
# AND it has been open for more than 30 days at the time of the email.
# This is stricter than the original binary check and avoids flagging
# tickets that are simply new and in-progress.
days_since_creation = (df['email_date'] - df['created']).dt.days
df['is_stalled'] = np.where(
    (~df['status'].isin(['Resolved', 'Closed'])) & (days_since_creation > 30),
    1, 0
)

# 4. Feature: subject_length
# Word count of email subject. Hypothesis: stressed developers write shorter subjects.
df['subject_length'] = df['email_subject'].astype(str).apply(lambda x: len(x.split()))

# 5. Feature: priority_numeric
# Convert categorical priority to ordinal numeric for ML compatibility.
priority_map = {'Blocker': 5, 'Critical': 4, 'Major': 3, 'Minor': 2, 'Trivial': 1}
df['priority_numeric'] = df['priority'].map(priority_map).fillna(0)


# =============================================================================
# ADVANCED FEATURE ENGINEERING (Socio-Technical Signals)
# =============================================================================

# Feature 6: sentiment_variance_per_ticket
# Measures polarisation within a ticket's email thread.
# High variance = some people calm, others stressed = conflict signal.
print("Calculating Sentiment Variance per Ticket...")
sentiment_stats = df.groupby('ticket_key')['behavior_score'].agg(['mean', 'std']).reset_index()
sentiment_stats.columns = ['ticket_key', 'avg_sentiment', 'sentiment_variance']
df = df.merge(sentiment_stats, on='ticket_key', how='left')
df['sentiment_variance'] = df['sentiment_variance'].fillna(0)

# Feature 7: email_volume_per_ticket
# Number of emails referencing this ticket.
# High volume = ticket is generating significant discussion (friction proxy).
print("Calculating Email Volume per Ticket...")
email_volume = df.groupby('ticket_key')['email_date'].count().reset_index()
email_volume.columns = ['ticket_key', 'email_volume_per_ticket']
df = df.merge(email_volume, on='ticket_key', how='left')

# Feature 8: sentiment_trend
# Difference between late-thread and early-thread average sentiment.
# Negative value = communication is deteriorating over time (early warning signal).
print("Calculating Temporal Sentiment Trend...")

def compute_sentiment_trend(group):
    group = group.sort_values('email_date')
    mid = len(group) // 2
    if mid == 0:
        return 0.0
    early_avg = group.iloc[:mid]['behavior_score'].mean()
    late_avg = group.iloc[mid:]['behavior_score'].mean()
    return float(late_avg - early_avg)

trend = df.groupby('ticket_key', group_keys=False).apply(
    lambda g: pd.Series({'sentiment_trend': compute_sentiment_trend(g)})
).reset_index()
df = df.merge(trend, on='ticket_key', how='left')

print(f"\nFeature Engineering Complete.")
print(f"   Total features engineered: {len(df.columns)}")
print(f"   Total records: {len(df)}")
print("\nSample of advanced features:")
print(df[['ticket_key', 'sentiment_variance', 'email_volume_per_ticket', 'sentiment_trend']].head(5))

# Summary statistics for report
print("\n--- Feature Summary Statistics ---")
for col in ['behavior_score', 'days_to_resolve', 'subject_length',
            'sentiment_variance', 'email_volume_per_ticket', 'sentiment_trend']:
    if col in df.columns:
        series = df[col].dropna()
        print(f"  {col}: mean={series.mean():.3f}, std={series.std():.3f}, "
              f"min={series.min():.3f}, max={series.max():.3f}")

stalled_pct = df['is_stalled'].mean() * 100
print(f"\n  is_stalled: {stalled_pct:.1f}% of records flagged as stalled")

df.to_csv(output_file, index=False)
print(f"\nEnriched dataset saved to: {output_file}")