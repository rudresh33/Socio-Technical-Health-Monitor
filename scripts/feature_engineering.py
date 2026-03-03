import pandas as pd
import numpy as np

# --- CONFIGURATION ---
input_file = "data/master_project_dataset.csv"
output_file = "data/enriched_project_dataset.csv"

print("Loading Master Dataset...")
df = pd.read_csv(input_file)

print("Starting Feature Engineering...")

# 1. Convert strings to datetime objects safely
df['created'] = pd.to_datetime(df['created'], errors='coerce', utc=True)
df['resolutiondate'] = pd.to_datetime(df['resolutiondate'], errors='coerce', utc=True)
df['email_date'] = pd.to_datetime(df['email_date'], errors='coerce', utc=True)

# Drop rows with unparseable crucial dates
df = df.dropna(subset=['created', 'email_date'])

# 2. Basic temporal metrics
df['days_to_resolve'] = (df['resolutiondate'] - df['created']).dt.days

# 3. Refined 'is_stalled' (Flags open tasks older than 30 days)
days_since_creation = (df['email_date'] - df['created']).dt.days
df['is_stalled'] = np.where(
    (~df['status'].isin(['Resolved', 'Closed'])) & (days_since_creation > 30), 
    1, 0
)

# 4. Feature: 'subject_length'
df['subject_length'] = df['email_subject'].astype(str).apply(lambda x: len(x.split()))

# 5. Feature: 'priority_level' 
priority_map = {'Blocker': 5, 'Critical': 4, 'Major': 3, 'Minor': 2, 'Trivial': 1}
df['priority_numeric'] = df['priority'].map(priority_map).fillna(0)

# --- ADVANCED FEATURE ENGINEERING ---

# Feature 1: sentiment_variance_per_ticket
print("Calculating Sentiment Variance...")
sentiment_stats = df.groupby('ticket_key')['behavior_score'].agg(['mean', 'std']).reset_index()
sentiment_stats.columns = ['ticket_key', 'avg_sentiment', 'sentiment_variance']
df = df.merge(sentiment_stats, on='ticket_key', how='left')
df['sentiment_variance'] = df['sentiment_variance'].fillna(0)

# Feature 2: email_volume_per_ticket
print("Calculating Email Volume Velocity...")
email_volume = df.groupby('ticket_key')['email_date'].count().reset_index()
email_volume.columns = ['ticket_key', 'email_volume_per_ticket']
df = df.merge(email_volume, on='ticket_key', how='left')

# Feature 3: sentiment_trend (Early vs Late)
print("Calculating Temporal Sentiment Trend...")
def compute_sentiment_trend(group):
    group = group.sort_values('email_date')
    mid = len(group) // 2
    if mid == 0:
        return 0
    early_avg = group.iloc[:mid]['behavior_score'].mean()
    late_avg = group.iloc[mid:]['behavior_score'].mean()
    return late_avg - early_avg  # Negative means deteriorating

trend = df.groupby('ticket_key').apply(compute_sentiment_trend).reset_index(name='sentiment_trend')
df = df.merge(trend, on='ticket_key', how='left')

print(f"✅ Engineered {len(df.columns)} features.")
print("Sample of new features:")
print(df[['ticket_key', 'sentiment_variance', 'email_volume_per_ticket', 'sentiment_trend']].head())

df.to_csv(output_file, index=False)
print(f"\n💾 Enriched dataset saved to: {output_file}")