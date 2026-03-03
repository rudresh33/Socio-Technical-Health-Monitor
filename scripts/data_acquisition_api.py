import requests
import os

# --- CONFIGURATION ---
YEARS = [2023, 2024]  # Changed to a list so it processes both automatically
OUTPUT_DIR = "data/mbox_files" # Saves inside the data folder automatically

TARGET_LISTS = [
    "common-dev",
    "hdfs-dev",
    "yarn-dev",
    "mapreduce-dev",
]
DOMAIN = "hadoop.apache.org"
BASE_URL = "https://lists.apache.org/api/mbox.lua"

def download_list_year(list_name, year):
    filename = f"{list_name}-{year}.mbox"
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\nProcessing {list_name} for {year}...")
    
    with open(filepath, 'wb') as outfile:
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}"
            url = f"{BASE_URL}?list={list_name}&domain={DOMAIN}&date={date_str}"
            print(f"  - Downloading {date_str}...", end=" ")
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=8192):
                        outfile.write(chunk)
                    print("Done.")
                else:
                    print(f"Failed (Status: {response.status_code})")
            except Exception as e:
                print(f"Error: {e}")
    print(f"✅ Saved: {filepath}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Starting downloads for Years: {YEARS}")
    
    # Loop through each year in the YEARS list
    for year in YEARS:
        for lst in TARGET_LISTS:
            download_list_year(lst, year)