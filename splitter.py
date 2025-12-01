import pandas as pd
import os
from tqdm import tqdm
import sys

# --- Configuration ---
# The directory containing all the individual CSV files
INPUT_DIR = "E:/CS/networks/Project/test_logs"
# The name of the resulting merged CSV file
OUTPUT_FILE = "test_downsampled_1of6.csv" # Changed output name to reflect downsampling
BENIGN_LABEL = 'BENIGN'
DOWNSAMPLE_RATE = 6 # We will save 1 row for every DOWNSAMPLE_RATE rows read


def merge_csv_files(input_directory, output_file, downsample_rate):
    """
    Reads all CSV files, concatenates them, downsamples the data by the 
    specified rate (1/rate), and saves the result to the output file.
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

    # 1. Load all files
    for filename in tqdm(csv_files, desc="Merging Files"):
        file_path = os.path.join(input_directory, filename)
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            all_data_frames.append(df)
            
        except Exception as e:
            tqdm.write(f"Warning: Skipping {filename} due to read error: {e}")

    if not all_data_frames:
        print("Error: No data could be loaded. Merge aborted.")
        sys.exit(1)

    # 2. Concatenate into a single DataFrame
    print("\nConcatenating DataFrames...")
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    initial_rows = len(combined_df)

    # 3. Downsample the DataFrame
    print(f"Downsampling data (1 out of every {downsample_rate} rows)...")
    
    # Use Pandas iloc to slice the DataFrame, selecting every Nth row
    downsampled_df = combined_df.iloc[::downsample_rate]
    
    final_rows = len(downsampled_df)
    
    # 4. Save the combined DataFrame to the new file
    print(f"Saving downsampled data to {output_file}...")
    downsampled_df.to_csv(output_file, index=False)
    
    print("-" * 50)
    print(f"SUCCESS: Merged {total_files} files into {output_file}")
    print(f"Initial rows: {initial_rows}")
    print(f"Final downsampled rows: {final_rows}")
    print("-" * 50)


if __name__ == "__main__":
    # Ensure pandas and tqdm are installed
    try:
        merge_csv_files(INPUT_DIR, OUTPUT_FILE, DOWNSAMPLE_RATE)
    except NameError:
        print("\nFATAL ERROR: Pandas or tqdm is not installed.")
        print("Please run: pip install pandas tqdm")