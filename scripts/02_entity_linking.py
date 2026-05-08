import mailbox
import pandas as pd
import re
import nltk
import os
import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- SETUP VADER ---
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# =============================================================================
# EXPANDED IT-DOMAIN LEXICON
# Purpose: Prevent standard VADER from misclassifying technical terminology
# as hostile language. Words are assigned domain-appropriate sentiment weights.
# =============================================================================
custom_lexicon = {
    # Priority labels — neutral in IT context
    'blocker': 0,
    'critical': -0.2,       # Urgency, not emotional crisis
    # Crash / failure terms — technical, not emotional
    'fatal': -0.5,
    'crash': -0.5,
    'panic': -0.4,
    'deadlock': -0.3,
    'corrupt': -0.5,
    'outage': -0.5,
    'incident': -0.2,
    'breach': -0.5,
    # Process control — neutral operational commands
    'kill': 0,
    'abort': 0,
    'dead': 0,              # "dead code" is neutral
    'terminate': 0,
    # Quality / stability terms — mildly negative
    'bug': -0.1,
    'defect': -0.2,
    'flaw': -0.2,
    'regression': -0.4,
    'broken': -0.3,
    'stuck': -0.3,
    'hanging': -0.3,
    'frozen': -0.3,
    'unstable': -0.3,
    'unresponsive': -0.3,
    'failing': -0.3,
    'fail': -0.3,
    'failure': -0.4,
    'error': -0.2,
    # Performance terms
    'leak': -0.4,
    'bottleneck': -0.2,
    'slow': -0.2,
    'timeout': -0.2,
    'latency': -0.1,
    'delay': -0.2,
    # Urgency / process terms
    'urgent': -0.3,
    'vulnerability': -0.5,
    'severity': -0.1,
    'warn': -0.1,
    'exception': -0.1,
    'drop': -0.1,
    'revert': -0.2,
    # Neutral development terms
    'workaround': 0,
    'hack': -0.1,           # Technical shortcut, not hostile
    # Positive resolution signals
    'patch': 0.1,
    'resolved': 0.2,
    'fixed': 0.3,
}
sia.lexicon.update(custom_lexicon)

# --- DIRECTORY CONFIGURATION ---
jira_csv_path = 'data/raw/issues.csv'
mbox_folder = 'data/raw/mbox_files'
output_folder = 'data/interim/parsed_chunks'


# FIX: Strip non-author text before sentiment scoring ---
# PROBLEM: Emails contain quoted replies (> lines), signatures, code blocks,
# stack traces, and log output. VADER was scoring ALL of this text, meaning
# a frustrated developer's score would be diluted by polite quoted replies
# from previous senders, or skewed by technical log output.
# SOLUTION: Strip everything that isn't the current sender's own words.
def strip_quoted_text(text):
    """Remove quoted replies, signatures, code blocks, and stack traces.
    Returns only the current sender's original text for accurate sentiment scoring."""
    lines = str(text).split('\n')
    clean_lines = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        # Skip quoted replies from previous emails
        if stripped.startswith('>'):
            continue
        # Stop at email signature
        if stripped == '--':
            break
        # Skip code blocks (fenced with ```)
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        # Skip Java/Python stack traces
        if stripped.startswith('at ') and '(' in stripped:
            continue
        # Skip build log lines
        if stripped.startswith('[INFO]') or stripped.startswith('[ERROR]') or stripped.startswith('[WARNING]'):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)


def get_sentiment(text):
    """Returns VADER compound sentiment score (-1 to +1)."""
    return sia.polarity_scores(text)['compound']


def parse_mbox_robust(mbox_path):
    """
    Scans an mbox file for JIRA ticket references using a greedy regex
    that handles punctuation variants: [HDFS-123], HDFS-123:, Re: HDFS-123.
    Returns a DataFrame of (ticket_key, email_subject, email_date, behavior_score, sender, body_snippet).
    """
    mbox = mailbox.mbox(mbox_path)
    email_data = []

    # Greedy non-capturing group — captures full ticket ID including number
    ticket_pattern = re.compile(r'(?:HADOOP|HDFS|YARN|MAPREDUCE)-\d+')

    count = 0
    for message in mbox:
        count += 1
        subject = message['subject'] or ""
        
        # --- EXTRACT SENDER ---
        sender = message.get('From', 'Unknown')
        
        try:
            payload = message.get_payload()
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == 'text/plain':
                        payload = part.get_payload(decode=True)
                        break
            body_text = str(payload)
        except Exception:
            body_text = ""

        # --- EXTRACT BODY SNIPPET ---
        # --- FIX: Store clean text snippet, not raw bytes with b'...' prefix ---
        body_snippet = strip_quoted_text(body_text)[:500].replace('\n', ' ').strip()

        full_text = f"{subject} {body_text}"
        mentioned_tickets = set(ticket_pattern.findall(full_text))

        # --- FIX: Score only the sender's own words, not quoted text ---
        cleaned_text = strip_quoted_text(body_text)
        sentiment = get_sentiment(cleaned_text[:1000])
        date_str = message['date']

        for ticket in mentioned_tickets:
            email_data.append({
                'ticket_key': ticket.upper().strip(),
                'email_subject': subject[:100],
                'email_date': date_str,
                'behavior_score': sentiment,
                # Add new fields to the output
                'sender': sender,
                'body_snippet': body_snippet
            })

    print(f"    -> Scanned {count} emails.")
    return pd.DataFrame(email_data)


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load JIRA data once — avoids reloading 1.1M rows for each mbox file
    print("Loading JIRA Data (This takes a moment)...")
    jira_df = pd.read_csv(jira_csv_path, low_memory=False)
    jira_df.columns = jira_df.columns.str.lower().str.strip()

    # Safe rename — only renames columns that exist, avoids silent failures
    rename_map = {
        'issue key': 'key',
        'status.name': 'status',
        'priority.name': 'priority'
    }
    # Only rename 'resolution' to 'resolutiondate' if resolutiondate doesn't already exist
    if 'resolution' in jira_df.columns and 'resolutiondate' not in jira_df.columns:
        rename_map['resolution'] = 'resolutiondate'

    jira_df.rename(columns=rename_map, inplace=True)
    jira_df['key'] = jira_df['key'].astype(str).str.upper().str.strip()
    print(f"JIRA Data Loaded: {len(jira_df)} tickets.\n")

    mbox_files = glob.glob(os.path.join(mbox_folder, "*.mbox"))
    print(f"Found {len(mbox_files)} mailing list archives to process.\n")

    for mbox_path in mbox_files:
        base_name = os.path.basename(mbox_path).replace('.mbox', '')
        output_csv_name = os.path.join(output_folder, f"parsed_{base_name}.csv")

        print(f"Processing: {base_name}...")
        email_df = parse_mbox_robust(mbox_path)

        if not email_df.empty:
            socio_technical_df = pd.merge(
                email_df, jira_df,
                left_on='ticket_key', right_on='key',
                how='inner'
            )
            if not socio_technical_df.empty:
                socio_technical_df.to_csv(output_csv_name, index=False)
                print(f"  SUCCESS: {len(socio_technical_df)} linked records saved to {output_csv_name}\n")
            else:
                print(f"  WARNING: No matching JIRA tickets found for {base_name}.\n")
        else:
            print(f"  WARNING: No ticket references found in {base_name}.\n")

    print("ALL FILES PROCESSED SUCCESSFULLY!")