import torch
import os
import time
import json 
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from tqdm import tqdm 

# --- NOTE: System Metric Imports (Install psutil/pynvml if needed) ---
# import psutil 
# try:
#     import pynvml
#     NVML_AVAILABLE = True
# except ImportError:
#     NVML_AVAILABLE = False
# SYSTEM_METRICS = [] # Global list to store system metrics

# --- 1. Configuration (GLOBAL VARIABLES) ---
LOCAL_MODEL_PATH = "E:/CS/networks/Project/local_model_files" 
BERT_BASE_TOKENIZER = "bert-base-uncased" 
TEST_LOGS_DIR = "E:/CS/networks/Project/test_logs" 

BATCH_SIZE = 256 
OUTPUT_FILE = "classification_output.jsonl" 
METRICS_REPORT_FILE = "benchmark_report.json" # <--- CHANGED TO JSON


# --- NEW: THRESHOLD CONFIGURATION ---
CLASSIFICATION_THRESHOLD = 0.85


# --- Argument Parsing Function ---
def parse_args():
    """Parses command line arguments for device selection."""
    parser = argparse.ArgumentParser(description="Run network log model benchmark.")
    parser.add_argument(
        '-d', '--device', 
        type=str, 
        default='auto', 
        choices=['auto', 'cpu', 'cuda', 'nvidia', 'radeon', 'amd', 'apple', 'mps'],
        help="Specify the target device: CUDA, MPS, CPU, or 'auto' (default)."
    )
    # Add argument to override the classification threshold from the command line
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=CLASSIFICATION_THRESHOLD, 
        help=f"Classification threshold (default: {CLASSIFICATION_THRESHOLD}). Probability >= threshold is MALICIOUS."
    )
    return parser.parse_args()


# --- 2. Device Setup (Defines the Global 'DEVICE') ---
ARGS = parse_args()
TARGET_DEVICE = ARGS.device.lower()
DEVICE = torch.device("cpu") # Initialize to CPU as a default fallback
DTYPE = torch.float32

# Update the global threshold based on command line input
CLASSIFICATION_THRESHOLD = ARGS.threshold

print(f"\n--- Device Setup ---")

# Check for specific GPU requests (NVIDIA/CUDA)
if (TARGET_DEVICE in ['cuda', 'nvidia']) and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # For RTX 2080 (Compute Capability >= 7)
    DTYPE = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
    print(f"Device: FORCED CUDA ({torch.cuda.get_device_name(0)})")

# Check for Apple Silicon requests (MPS)
elif (TARGET_DEVICE in ['apple', 'mps']) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.bfloat16
    print("Device: FORCED Apple Silicon (MPS)")

# Check for CPU/AMD/RADEON requests OR if CUDA/MPS were requested but unavailable
elif (TARGET_DEVICE in ['cpu', 'amd', 'radeon']) or (TARGET_DEVICE != 'auto'):
    # This handles explicit CPU request AND fallback if the requested GPU is not found
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print("Device: FORCED CPU/Fallback")

# Handle 'auto' mode (default behavior)
elif TARGET_DEVICE == 'auto':
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DTYPE = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
        print(f"Device: AUTO CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DTYPE = torch.bfloat16
        print("Device: AUTO Apple Silicon (MPS)")
    else:
        DEVICE = torch.device("cpu")
        print("Device: AUTO CPU")

print(f"Using Data Type: {DTYPE}")
print(f"Using Classification Threshold: {CLASSIFICATION_THRESHOLD}")
# -------------------------------------------------------------

# --- 3. Preprocessing Functions ---
FEATURE_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 
    'Flow Packets/s', 'Fwd PSH Flags', 'FIN Flag Count', 'SYN Flag Count', 
    'ACK Flag Count', 'URG Flag Count', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]

def format_row_to_sentence(row):
    """Converts a DataFrame row of network features into a descriptive sentence."""
    parts = []
    parts.append(f"dest_port is {row['Destination Port']}")
    parts.append(f"protocol is tcp") 
    parts.append(f"flow_duration is {row['Flow Duration']}")
    parts.append(f"fwd_pkts is {row['Total Fwd Packets']}")
    parts.append(f"bwd_pkts is {row['Total Backward Packets']}")
    parts.append(f"fwd_bytes is {row['Total Length of Fwd Packets']}")
    parts.append(f"bwd_bytes is {row['Total Length of Bwd Packets']}")
    parts.append(f"flow_rate_bytes is {row['Flow Bytes/s']:.0f}")
    parts.append(f"flow_rate_pkts is {row['Flow Packets/s']:.0f}")
    if row['SYN Flag Count'] > 0: parts.append("SYN_flag_set")
    if row['ACK Flag Count'] > 0: parts.append("ACK_flag_set")
    if row['FIN Flag Count'] > 0: parts.append("FIN_flag_set")
    if row['Fwd PSH Flags'] > 0: parts.append("PSH_flag_set")
    parts.append(f"fwd_win_bytes is {row['Init_Win_bytes_forward']}")
    parts.append(f"bwd_win_bytes is {row['Init_Win_bytes_backward']}")
    
    return ". ".join(parts) + "."

def load_and_preprocess_data():
    """Loads all CSVs, extracts features, and creates the list of sentences, timing the process."""
    print(f"\n--- Loading and Preprocessing Data from {TEST_LOGS_DIR} ---")
    start_time = time.time()
    all_data_frames = []
    
    try:
        csv_files = [f for f in os.listdir(TEST_LOGS_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: Directory not found at {TEST_LOGS_DIR}")
        return [], [], 0.0

    for filename in tqdm(csv_files, desc="Loading CSV Files"): 
        file_path = os.path.join(TEST_LOGS_DIR, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            
            if not all(col in df.columns for col in FEATURE_COLUMNS) or 'Label' not in df.columns:
                tqdm.write(f"Warning: Skipping {filename}. Missing required columns or 'Label'.")
                continue
            
            all_data_frames.append(df)
        except Exception as e:
            tqdm.write(f"Error reading {filename}: {e}")
    
    if not all_data_frames:
        print("Error: No valid CSV files were loaded.")
        return [], [], 0.0
        
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    text_logs = combined_df.apply(format_row_to_sentence, axis=1).tolist()
    ground_truth_labels = combined_df['Label'].tolist()
    
    end_time = time.time()
    preprocessing_time = end_time - start_time

    print(f"Total rows loaded: {len(combined_df)}")
    print(f"Preprocessing time: {preprocessing_time:.4f} seconds")
    
    return text_logs, ground_truth_labels, preprocessing_time


# --- UPDATED: Function to Write Performance Report to JSON (.json) ---
def write_performance_report(metrics_data):
    """Writes the gathered performance metrics and timing to a structured JSON file."""
    try:
        # We write the metrics_data dictionary directly as JSON
        with open(METRICS_REPORT_FILE, 'w') as f:
            # We use indent=4 for human readability
            json.dump(metrics_data, f, indent=4)
            
        print(f"\nPerformance metrics saved to {METRICS_REPORT_FILE}")
    except Exception as e:
        print(f"Error writing metrics report: {e}")


# --- 4. Main Benchmark Function ---

def load_and_run_benchmark():
    """Loads resources and performs the batch stress test, writing results to .jsonl."""
    
    TEST_LOGS, GROUND_TRUTH_LABELS, PREPROCESS_TIME = load_and_preprocess_data()
    NUM_SAMPLES = len(TEST_LOGS)
    
    if NUM_SAMPLES == 0:
        return 

    # 4a. Load Tokenizer & Model
    try:
        print(f"\n--- Loading Tokenizer from Hub: {BERT_BASE_TOKENIZER} ---")
        tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_TOKENIZER)
    except Exception as e:
        print(f"Error loading tokenizer: {e}"); return

    try:
        print(f"Loading Model from Local Path: {LOCAL_MODEL_PATH}")
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
        
        # --- FIX 1: Override Generic Labels ---
        model.config.id2label = {0: "BENIGN", 1: "MALICIOUS"}
        label_map = model.config.id2label
        # --- END FIX 1 ---
        
        if DTYPE != torch.float32: model = model.to(DTYPE)
        model.to(DEVICE); model.eval() 
    except Exception as e:
        print(f"Error loading model: {e}"); return


    # 4b. Benchmark Loop Setup
    total_logs_processed = 0
    total_time_taken = 0.0
    print(f"\n--- Starting Stress Test: {NUM_SAMPLES} logs, Batch Size {BATCH_SIZE} ---")
    print(f"Classification Threshold set to: {CLASSIFICATION_THRESHOLD}")


    with open(OUTPUT_FILE, 'w') as outfile:
        autocast_context = torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=(DTYPE != torch.float32))
        
        num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
        
        # FIX: Assign tqdm iterable to 'pbar' to call its methods correctly
        pbar = tqdm(range(0, NUM_SAMPLES, BATCH_SIZE), total=num_batches, desc="Running Inference")

        # 4c. Main Batch Processing Loop
        for i in pbar: 
            
            # (Optional: call log_system_metrics() here)
            
            batch_logs = TEST_LOGS[i:i + BATCH_SIZE]
            batch_ground_truth = GROUND_TRUTH_LABELS[i:i + BATCH_SIZE]
            
            # Tokenize and move to device
            inputs = tokenizer(batch_logs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # --- TIMING & INFERENCE ---
            if DEVICE.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
                
            with torch.no_grad():
                with autocast_context:
                    outputs = model(**inputs)
            
            if DEVICE.type == 'cuda':
                end_event.record(); torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                batch_time = time.time() - start_time
                
            total_logs_processed += len(batch_logs)
            total_time_taken += batch_time
            
            # 4d. Process Predictions and Write to .jsonl
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get the confidence for MALICIOUS (index 1)
            probabilities_cpu = probabilities.cpu().numpy()
            malicious_confidence = probabilities_cpu[:, 1]
            
            # --- NEW THRESHOLD PREDICTION LOGIC ---
            # Predict 1 (MALICIOUS) if confidence >= threshold, else 0 (BENIGN)
            predicted_ids = (malicious_confidence >= CLASSIFICATION_THRESHOLD).astype(int)
            # --- END NEW THRESHOLD PREDICTION LOGIC ---
            
            for j, log_text in enumerate(batch_logs):
                pred_id = predicted_ids[j]
                
                # FIX 2: pred_label now uses the correctly overridden label_map
                pred_label = label_map.get(pred_id, "UNKNOWN")
                
                result = {
                    "flow_id": i + j,
                    "log_input": log_text,
                    "prediction": pred_label, # Will now be BENIGN or MALICIOUS
                    "confidence": float(malicious_confidence[j]),
                    "ground_truth": batch_ground_truth[j]
                }
                outfile.write(json.dumps(result) + '\n')
            
            # Update the progress bar description
            if total_logs_processed > 0 and total_time_taken > 0:
                current_throughput = total_logs_processed / total_time_taken
                pbar.set_postfix_str(f"Throughput: {current_throughput:.0f} logs/s")


    # 4e. Final Metric Calculation and Reporting
    total_pipeline_time = PREPROCESS_TIME + total_time_taken
    
    # Calculate final metrics
    throughput = total_logs_processed / total_time_taken if total_time_taken > 0 else 0
    average_latency = (total_time_taken / total_logs_processed) * 1000 if total_logs_processed > 0 else 0

    # 5. Compile metrics dictionary for reporting
    metrics_data = {
        'DEVICE': str(DEVICE),
        'DTYPE': str(DTYPE),
        'TOTAL_LOGS': total_logs_processed,
        'PREPROCESS_TIME': PREPROCESS_TIME,
        'INFERENCE_TIME': total_time_taken,
        'TOTAL_PIPELINE_TIME': total_pipeline_time,
        'THROUGHPUT': throughput,
        'LATENCY': average_latency
    }
    
    # Output to console
    print("\n--- Performance Report ---")
    print(f"Hardware Tested: {metrics_data['DEVICE']} (DType: {metrics_data['DTYPE']})")
    print(f"Total Logs Processed: {metrics_data['TOTAL_LOGS']}")
    print(f"\n--- Timing Breakdown ---")
    print(f"1. Preprocessing Time (I/O & Formatting): {metrics_data['PREPROCESS_TIME']:.4f} seconds")
    # FIX: Corrected dict key reference here
    print(f"2. Inference Time (Model Execution): {metrics_data['INFERENCE_TIME']:.4f} seconds") 
    print(f"Total Pipeline Time: {metrics_data['TOTAL_PIPELINE_TIME']:.4f} seconds")
    if throughput > 0:
        print(f"\n--- Execution Metrics ---")
        print(f"**Throughput (logs/sec): {metrics_data['THROUGHPUT']:.2f}**")
        print(f"**Average Latency (ms/log): {metrics_data['LATENCY']:.4f}**")

    # Output to file
    write_performance_report(metrics_data)
    
    print(f"\n\n--- EVALUATION REQUIRED ---")
    print(f"Run the evaluation script to get model quality metrics:")
    print(f"python evaluate.py --file {OUTPUT_FILE}")


if __name__ == "__main__":
    load_and_run_benchmark()