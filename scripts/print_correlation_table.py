"""
Prints the top 10 strongest pairwise Pearson correlations between
communication features in the ticket-level dataset.

Output is pipe-separated so it can be pasted into Word and converted
via "Convert Text to Table" with "|" as the column separator.
"""

import pandas as pd

INPUT_FILE = "data/processed/isa3_ticket_level.csv"

FEATURES = [
    "email_count_per_ticket",
    "subject_length",
    "avg_sentiment",
    "sentiment_variance",
    "sentiment_trend",
    "priority_numeric",
    "unique_senders",
    "description_length",
]

df = pd.read_csv(INPUT_FILE, low_memory=False)
corr = df[FEATURES].corr()

# Extract unique upper-triangle pairs (i < j avoids diagonal + duplicates)
pairs = []
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        f1, f2 = FEATURES[i], FEATURES[j]
        pairs.append((f1, f2, corr.loc[f1, f2]))

pairs.sort(key=lambda p: abs(p[2]), reverse=True)
top10 = pairs[:10]

# Print header + rows
print(f"| {'Feature 1':<25} | {'Feature 2':<25} | {'Correlation':>11} |")
print(f"| {'-' * 25} | {'-' * 25} | {'-' * 11} |")
for f1, f2, r in top10:
    print(f"| {f1:<25} | {f2:<25} | {r:>+11.3f} |")

