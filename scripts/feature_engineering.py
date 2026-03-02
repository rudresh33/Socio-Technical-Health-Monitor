import pandas as pd
import numpy as np

# --- CONFIGURATION ---
input_file = "data/master_project_dataset.csv"
output_file = "data/enriched_project_dataset.csv"

print("Loading Master Dataset...")
df = pd.read_csv(input_file)

print("Starting Feature Engineering...")

# 1. Convert strings to datetime objects
# The errors='coerce' handles any weird formatting safely
df['created'] = pd.to_datetime(df['created'], errors='coerce', utc=True)
df['resolutiondate'] = pd.to_datetime(df['resolutiondate'], errors='coerce', utc=True)
df['email_date'] = pd.to_datetime(df['email_date'], errors='coerce', utc=True)

# 2. Feature: 'days_to_resolve'
# How long did the task take to finish? (NaN if it is still Open)
df['days_to_resolve'] = (df['resolutiondate'] - df['created']).dt.days

# 3. Feature: 'is_stalled'
# If it's still Open after 30 days, we flag it as a stalled/risk task
df['is_stalled'] = np.where((df['status'] != 'Resolved') & (df['status'] != 'Closed'), 1, 0)

# 4. Feature: 'subject_length'
# Hypothesis: Stressed developers write shorter, more urgent subjects
df['subject_length'] = df['email_subject'].astype(str).apply(lambda x: len(x.split()))

# 5. Feature: 'priority_level' (Numeric)
# ML models prefer numbers over text categories
priority_map = {'Blocker': 5, 'Critical': 4, 'Major': 3, 'Minor': 2, 'Trivial': 1}
df['priority_numeric'] = df['priority'].map(priority_map).fillna(0)

# Drop any rows where we couldn't parse the dates properly
df = df.dropna(subset=['created', 'email_date'])

print(f"✅ Engineered {len(df.columns)} features.")
print("Sample of new features:")
print(df[['ticket_key', 'days_to_resolve', 'is_stalled', 'subject_length']].head())

df.to_csv(output_file, index=False)
print(f"\n💾 Enriched dataset saved to: {output_file}")