import pandas as pd
import os
import glob

# --- CONFIGURATION ---
input_folder = "data/parsed_chunks"
output_file = "data/master_project_dataset.csv"

all_files = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"Found {len(all_files)} datasets. Merging...")

df_list = []
for file in all_files:
    print(f" - Loading {os.path.basename(file)}...")
    df = pd.read_csv(file)
    df_list.append(df)

master_df = pd.concat(df_list, ignore_index=True)
initial_len = len(master_df)

# =============================================================================
# BUG FIX: Parse raw string dates to standardized UTC before deduplicating.
# Raw mbox date strings can have different timezone formats across files
# (e.g., "+0000", "GMT", "UTC") which would cause identical emails to appear
# as unique rows if we deduplicate on the raw string directly.
# =============================================================================
print("Standardizing timestamps for accurate deduplication...")
master_df['email_date_parsed'] = pd.to_datetime(
    master_df['email_date'], errors='coerce', utc=True
)

master_df.drop_duplicates(subset=['ticket_key', 'email_date_parsed'], inplace=True)
master_df.drop(columns=['email_date_parsed'], inplace=True)

final_len = len(master_df)

print(f"\nMerging Complete!")
print(f"   Total Rows Before Deduplication: {initial_len}")
print(f"   Total Unique Socio-Technical Rows: {final_len}")
print(f"   Duplicates Removed: {initial_len - final_len}")

master_df.to_csv(output_file, index=False)
print(f"\nMaster Dataset saved to: {output_file}")