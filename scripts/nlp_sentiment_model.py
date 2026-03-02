import pandas as pd
from collections import Counter
import re
import os

input_file = "data/enriched_project_dataset.csv"

print("Loading Enriched Dataset...")
df = pd.read_csv(input_file, low_memory=False)

# Look at highly negative emails
negative_emails = df[df['behavior_score'] < -0.4]['email_subject'].astype(str)

print(f"\nAnalyzing {len(negative_emails)} highly stressed email subjects...")

all_words = []
for subject in negative_emails:
    clean_subject = re.sub(r'(?:HADOOP|HDFS|YARN|MAPREDUCE)-\d+', '', subject)
    clean_subject = re.sub(r'[^\w\s]', '', clean_subject).lower()
    
    words = clean_subject.split()
    
    # THE FIX: Added bot/automated jargon to the stop words
    stop_words = {
        # Standard English
        'the', 'a', 'to', 'in', 'of', 'for', 'is', 'on', 're', 'and', 'with', 'from', 'by', 'update',
        # Hadoop / JIRA standard words
        'hadoop', 'apache', 'jira', 'created', 'commented', 'resolved', 'patch',
        # BOT / Automated Build Jargon (The Noise)
        'report', 'qbt', 'linuxx86_64', 'trunkjdk11', 'trunkjdk8', 'branch33jdk8', 'build', 'failure'
    }
    
    words = [w for w in words if w not in stop_words and len(w) > 2]
    all_words.extend(words)

word_counts = Counter(all_words)

print("\n🚨 REFINED TOP 15 HUMAN STRESS/URGENCY KEYWORDS 🚨")
for word, count in word_counts.most_common(15):
    print(f" - {word}: {count} times")