import pandas as pd
import os
from tqdm import tqdm
import sys

# --- Configuration ---
# The directory containing all the individual CSV files
INPUT_DIR = "E:/CS/networks/Project/test_logs"
# The name of the resulting merged CSV file
OUTPUT_FILE = "test.csv"
BENIGN_LABEL = 'BENIGN'

def merge_csv_files(input_directory, output_file):
    """
    Reads all CSV files in the input directory, concatenates them into a single 
    DataFrame, and saves the result to the specified output file.
    """
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found at {input_directory}")
        sys.exit(1)

    all_data_frames = []
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    if total_files == 0:
        print(f"Error: No CSV files found in {input_directory}. Nothing to merge.")
        sys.exit(1)

    print(f"--- Starting Merge of {total_files} CSV files ---")

    # Use tqdm to show progress during file loading
    for filename in tqdm(csv_files, desc="Merging Files"):
        file_path = os.path.join(input_directory, filename)
        
        try:
            # Read the CSV file (low_memory=False is essential for large, complex datasets)
            df = pd.read_csv(file_path, low_memory=False)
            
            # Clean column names by stripping whitespace (a necessary step confirmed by your previous issues)
            df.columns = df.columns.str.strip()
            
            all_data_frames.append(df)
            
        except Exception as e:
            tqdm.write(f"Warning: Skipping {filename} due to read error: {e}")

    if not all_data_frames:
        print("Error: No data could be loaded. Merge aborted.")
        sys.exit(1)

    # Concatenate all DataFrames into a single DataFrame
    print("\nConcatenating DataFrames...")
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    # Save the combined DataFrame to the new file
    print(f"Saving merged data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print("-" * 50)
    print(f"SUCCESS: Merged {total_files} files into {output_file}")
    print(f"Total rows in {output_file}: {len(combined_df)}")
    print("-" * 50)


if __name__ == "__main__":
    # Ensure pandas and tqdm are installed
    try:
        merge_csv_files(INPUT_DIR, OUTPUT_FILE)
    except NameError:
        print("\nFATAL ERROR: Pandas or tqdm is not installed.")
        print("Please run: pip install pandas tqdm")