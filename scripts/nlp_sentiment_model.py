import pandas as pd
from collections import Counter
import re

# --- CONFIGURATION ---
input_file = "data/enriched_project_dataset.csv"

# =============================================================================
# STOP WORDS CONFIGURATION
# Split into categories for transparency and easy extension.
# This list should be updated if new bot patterns are discovered.
# =============================================================================
STOP_WORDS = set()

# Standard English stop words
STOP_WORDS.update({
    'the', 'a', 'to', 'in', 'of', 'for', 'is', 'on', 're', 'and',
    'with', 'from', 'by', 'update', 'at', 'as', 'an', 'it', 'its',
    'this', 'that', 'are', 'was', 'has', 'have', 'had', 'not', 'be',
    'been', 'will', 'can', 'may', 'all', 'also', 'into', 'or', 'but'
})

# Hadoop / JIRA process words (present in normal operational emails)
STOP_WORDS.update({
    'hadoop', 'apache', 'jira', 'created', 'commented', 'resolved',
    'patch', 'hdfs', 'yarn', 'mapreduce', 'commit', 'merge', 'review'
})

# CI/CD Bot Jargon — automated build emails that skew sentiment scores
# Identified through initial NLP run (first iteration of analysis)
STOP_WORDS.update({
    'report', 'qbt', 'linuxx86_64', 'trunkjdk11', 'trunkjdk8',
    'branch33jdk8', 'build', 'failure', 'linux', 'jdk', 'trunk',
    'branch', 'jdk8', 'jdk11', 'jdk17', 'x86', 'x64'
})


def extract_human_keywords(df, sentiment_threshold=-0.4, top_n=20):
    """
    Extracts the most frequent keywords from emails with sentiment
    below the given threshold. These represent genuine human stress signals
    after bot noise has been removed.
    """
    negative_emails = df[df['behavior_score'] < sentiment_threshold]['email_subject'].astype(str)
    print(f"Analyzing {len(negative_emails)} highly stressed email subjects "
          f"(behavior_score < {sentiment_threshold})...")

    all_words = []
    for subject in negative_emails:
        # Remove ticket IDs (e.g., HDFS-1234) — not human language
        clean = re.sub(r'(?:HADOOP|HDFS|YARN|MAPREDUCE)-\d+', '', subject)
        # Remove punctuation and lowercase
        clean = re.sub(r'[^\w\s]', '', clean).lower()
        words = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 2]
        all_words.extend(words)

    return Counter(all_words).most_common(top_n)


def cluster_keywords(keywords):
    """
    Groups keywords into interpretable clusters for the report.
    """
    release_keywords = {'release', 'vote', 'rc2', 'rc3', 'rc4', '335', '341', '340', 'candidate'}
    blockage_keywords = {'block', 'fix', 'fails', 'fail', 'error', 'wrong', 'broken', 'stuck'}
    urgency_keywords = {'when', 'should', 'due', 'not', 'please', 'still', 'need', 'must'}
    domain_keywords = {'s3a', 'hdfs', 'yarn', 'namenode', 'datanode', 'container', 'scheduler'}

    print("\n--- KEYWORD CLUSTERS ---")
    print("\n[Release Pressure Cluster]")
    release_found = [(w, c) for w, c in keywords if w in release_keywords]
    for w, c in release_found:
        print(f"  {w}: {c} times")

    print("\n[Technical Blockage Cluster]")
    blockage_found = [(w, c) for w, c in keywords if w in blockage_keywords]
    for w, c in blockage_found:
        print(f"  {w}: {c} times")

    print("\n[Urgency / Friction Cluster]")
    urgency_found = [(w, c) for w, c in keywords if w in urgency_keywords]
    for w, c in urgency_found:
        print(f"  {w}: {c} times")

    print("\n[Domain-Specific Pain Points]")
    domain_found = [(w, c) for w, c in keywords if w in domain_keywords]
    for w, c in domain_found:
        print(f"  {w}: {c} times")


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Loading Enriched Dataset...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Dataset loaded: {len(df)} records\n")

    keywords = extract_human_keywords(df, sentiment_threshold=-0.4, top_n=20)

    print("\n REFINED TOP 20 HUMAN STRESS/URGENCY KEYWORDS (Bot noise removed) ")
    for word, count in keywords:
        print(f"  - {word}: {count} times")

    cluster_keywords(keywords)

    print("\n NOTE FOR UNNAT:")
    print("  Add these findings to the ISA-II PPT under 'Linguistic Analysis of Developer Stress'.")
    print("  Explain that the first run revealed CI/CD bot keywords (qbt, linuxx86_64),")
    print("  which were then removed to isolate genuine human communication signals.")
    print("  This iterative cleaning process demonstrates real NLP domain adaptation.")