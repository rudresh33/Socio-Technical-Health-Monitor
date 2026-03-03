import pandas as pd
import os
import glob

# Path where Sarvadhnya's CSVs are stored
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

# Bug Fix: Parse raw string dates to standard UTC before deduplicating
print("Standardizing timestamps for accurate deduplication...")
master_df['email_date_parsed'] = pd.to_datetime(master_df['email_date'], errors='coerce', utc=True)

master_df.drop_duplicates(subset=['ticket_key', 'email_date_parsed'], inplace=True)
master_df.drop(columns=['email_date_parsed'], inplace=True) # Cleanup

final_len = len(master_df)

print(f"\n✅ Merging Complete!")
print(f"Total Rows Before Deduplication: {initial_len}")
print(f"Total Unique Socio-Technical Rows: {final_len}")

master_df.to_csv(output_file, index=False)
print(f"💾 Master Dataset saved to: {output_file}")