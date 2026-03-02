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
sia.lexicon.update({
    'blocker': 0, 'critical': -0.2, 'fatal': -0.5, 
    'kill': 0, 'dead': 0, 'exception': -0.1
})

# --- DIRECTORY CONFIGURATION ---
jira_csv_path = 'data/issues.csv' 
mbox_folder = 'data/mbox_files'
output_folder = 'data/parsed_chunks'

def get_sentiment(text):
    return sia.polarity_scores(text)['compound']

def parse_mbox_robust(mbox_path):
    mbox = mailbox.mbox(mbox_path)
    email_data = []
    ticket_pattern = re.compile(r'(?:HADOOP|HDFS|YARN|MAPREDUCE)-\d+')

    count = 0
    for message in mbox:
        count += 1
        subject = message['subject'] or ""
        try:
            payload = message.get_payload()
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == 'text/plain':
                        payload = part.get_payload(decode=True)
                        break
            body_text = str(payload)
        except:
            body_text = ""
        
        full_text = f"{subject} {body_text}"
        mentioned_tickets = set(ticket_pattern.findall(full_text))
        sentiment = get_sentiment(body_text[:1000])
        date_str = message['date']
        
        for ticket in mentioned_tickets:
            email_data.append({
                'ticket_key': ticket.upper().strip(), 
                'email_subject': subject[:100],
                'email_date': date_str,
                'behavior_score': sentiment
            })
    print(f"    -> Scanned {count} emails.")
    return pd.DataFrame(email_data)

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Load JIRA Data ONCE (Huge time saver)
    print("Loading JIRA Data (This takes a moment)...")
    jira_df = pd.read_csv(jira_csv_path, low_memory=False)
    jira_df.columns = jira_df.columns.str.lower().str.strip()
    rename_map = {'issue key': 'key', 'status.name': 'status', 'priority.name': 'priority', 'resolution': 'resolutiondate'}
    jira_df.rename(columns=rename_map, inplace=True)
    jira_df['key'] = jira_df['key'].astype(str).str.upper().str.strip()
    print("✅ JIRA Data Loaded.\n")

    # 2. Find all MBOX files automatically
    mbox_files = glob.glob(os.path.join(mbox_folder, "*.mbox"))
    print(f"Found {len(mbox_files)} mailing list archives to process.\n")

    # 3. Process each file in a loop
    for mbox_path in mbox_files:
        # Extract just the filename without the folder path or extension
        base_name = os.path.basename(mbox_path).replace('.mbox', '')
        output_csv_name = os.path.join(output_folder, f"parsed_{base_name}.csv")
        
        print(f"Processing: {base_name}...")
        
        # Parse the emails
        email_df = parse_mbox_robust(mbox_path)
        
        if not email_df.empty:
            # Merge with JIRA
            socio_technical_df = pd.merge(email_df, jira_df, left_on='ticket_key', right_on='key', how='inner')
            
            if not socio_technical_df.empty:
                socio_technical_df.to_csv(output_csv_name, index=False)
                print(f"  ✅ SUCCESS: {len(socio_technical_df)} linked records saved to {output_csv_name}\n")
            else:
                print(f"  ⚠️ No matching JIRA tickets found for {base_name}.\n")
        else:
            print(f"  ❌ No ticket references found in {base_name}.\n")
            
    print("🎉 ALL FILES PROCESSED SUCCESSFULLY!")