import requests
import os

# --- CONFIGURATION ---
# To download a different year range, update this list.
YEARS = [2023, 2024]

OUTPUT_DIR = "data/mbox_files"

# Developer mailing lists only — commits and issues lists are excluded
# because they contain automated bot messages with zero sentiment value.
TARGET_LISTS = [
    "common-dev",
    "hdfs-dev",
    "yarn-dev",
    "mapreduce-dev",
]

DOMAIN = "hadoop.apache.org"
BASE_URL = "https://lists.apache.org/api/mbox.lua"


def download_list_year(list_name, year):
    """
    Downloads all 12 monthly mbox archives for a given list and year,
    concatenates them into a single yearly .mbox file.
    Monthly granularity is the smallest unit available from Apache Archives.
    """
    filename = f"{list_name}-{year}.mbox"
    filepath = os.path.join(OUTPUT_DIR, filename)

    print(f"\nProcessing {list_name} for {year}...")

    with open(filepath, 'wb') as outfile:
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}"
            url = f"{BASE_URL}?list={list_name}&domain={DOMAIN}&date={date_str}"
            print(f"  - Downloading {date_str}...", end=" ", flush=True)
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=8192):
                        outfile.write(chunk)
                    print("Done.")
                else:
                    print(f"Failed (HTTP {response.status_code})")
            except requests.exceptions.Timeout:
                print("Timeout — skipped.")
            except Exception as e:
                print(f"Error: {e}")

    print(f"Saved: {filepath}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Starting downloads for Years: {YEARS}")
    print(f"Target lists: {TARGET_LISTS}\n")

    for year in YEARS:
        for lst in TARGET_LISTS:
            download_list_year(lst, year)

    print("\nAll downloads complete.")