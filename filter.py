import pandas as pd
import os
import sys

# --- Configuration ---
# The path to your test data directory
TEST_LOGS_DIR = "E:/CS/networks/Project/test_logs"
# TARGET_CSV_FILE is no longer needed as we iterate through the directory
# TARGET_CSV_FILE = "sample_test_flow.csv" 

#FILE_PATH = os.path.join(TEST_LOGS_DIR, TARGET_CSV_FILE) # This variable is no longer used, but kept for compatibility.
BENIGN_LABEL = 'BENIGN'

def find_malicious_labels_in_directory(directory_path):
    """
    Reads all CSV files in a directory, filters out BENIGN flows, 
    and prints the unique malicious labels found across the entire dataset.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    all_malicious_labels = set()
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    if total_files == 0:
        print(f"Error: No CSV files found in {directory_path}.")
        return

    print(f"--- Searching {total_files} CSV files for Malicious Labels ---")

    for filename in csv_files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, low_memory=False)
            
            # Clean column names (strip leading/trailing whitespace)
            df.columns = df.columns.str.strip()
            
            if 'Label' not in df.columns:
                print(f"Warning: Skipping {filename}. Column 'Label' not found.")
                continue

            # Convert the Label column to uppercase and strip whitespace for consistent filtering
            df['Label'] = df['Label'].str.strip().str.upper()
            
            # Filter for all rows where the Label is NOT BENIGN
            malicious_flows = df[df['Label'] != BENIGN_LABEL]
            
            # Add the unique non-BENIGN labels from this file to the master set
            all_malicious_labels.update(malicious_flows['Label'].unique())
            
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    # --- Final Output ---
    if not all_malicious_labels:
        print(f"Scan complete. No malicious labels found (all flows are '{BENIGN_LABEL}').")
    else:
        print(f"\n--- Unique Malicious Labels Found Across ALL {total_files} Files ---")
        sorted_labels = sorted(list(all_malicious_labels))
        for label in sorted_labels:
            print(f"- {label}")
        print("-" * 50)

if __name__ == "__main__":
    find_malicious_labels_in_directory(TEST_LOGS_DIR)