import pandas as pd
import os
import glob

# Path where Sarvadhnya's CSVs are stored
input_folder = "data/parsed_chunks"
output_file = "data/master_project_dataset.csv"

# Find all CSV files in the folder
all_files = glob.glob(os.path.join(input_folder, "*.csv"))

print(f"Found {len(all_files)} datasets. Merging...")

# Read and concatenate all files
df_list = []
for file in all_files:
    print(f" - Loading {os.path.basename(file)}...")
    df = pd.read_csv(file)
    df_list.append(df)

# Combine them all into one giant DataFrame
master_df = pd.concat(df_list, ignore_index=True)

# Drop exact duplicates (in case emails were cross-posted)
initial_len = len(master_df)
master_df.drop_duplicates(subset=['ticket_key', 'email_date'], inplace=True)
final_len = len(master_df)

print(f"\n✅ Merging Complete!")
print(f"Total Rows Before Deduplication: {initial_len}")
print(f"Total Unique Socio-Technical Rows: {final_len}")

# Save the master file
master_df.to_csv(output_file, index=False)
print(f"💾 Master Dataset saved to: {output_file}")