import json
import argparse
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np

# --- Configuration ---
MALICIOUS_LABEL = "MALICIOUS"
BENIGN_LABEL = "BENIGN"
LABELS_TO_USE = [BENIGN_LABEL, MALICIOUS_LABEL]

# All unique attack types provided by the user (CLEANED VERSION FOR COMPARISON)
SPECIFIC_ATTACKS = [
    "BOT", "DDOS", "DOS GOLDENEYE", "DOS HULK", "DOS SLOWHTTPTEST", 
    "DOS SLOWLORIS", "FTP-PATATOR", "HEARTBLEED", "INFILTRATION", 
    "PORTSCAN", "SSH-PATATOR", 
    "WEB ATTACK - BRUTE FORCE",  # Using clean hyphen for easier manipulation
    "WEB ATTACK - SQL INJECTION",
    "WEB ATTACK - XSS"          # FIXED typo: WEB ATTACK - XSS
]

def normalize_web_attack_label(raw_label):
    """
    Handles the common corrupted hyphen issue (e.g., 'WEB ATTACK  XSS')
    by cleaning the string based on the known format.
    """
    normalized_label = raw_label.upper().strip()
    
    # Check for the common WEB ATTACK prefix
    if normalized_label.startswith("WEB ATTACK"):
        # The known corrupted format is "WEB ATTACK <junk> ATTACK_TYPE"
        
        if "BRUTE FORCE" in normalized_label:
            return "WEB ATTACK - BRUTE FORCE"
        elif "SQL INJECTION" in normalized_label:
            return "WEB ATTACK - SQL INJECTION"
        elif "XSS" in normalized_label:
            return "WEB ATTACK - XSS"
            
    return normalized_label # Return original if it's not a known web attack corruption


def calculate_evaluation_metrics(jsonl_file):
    """
    Reads the JSON Lines output file, calculates overall binary metrics 
    and detailed per-attack confusion matrices.
    """
    print(f"\n--- Loading and Evaluating Results from {jsonl_file} ---")
    
    true_labels_binary = []       
    predicted_labels_binary = []  
    
    # Dictionary to store per-attack statistics (e.g., {'DDOS': {'TP': 0, 'FN': 0, ...}})
    per_attack_results = {attack: Counter({'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}) for attack in SPECIFIC_ATTACKS}
    
    total_entries = 0
    
    # Read the results from the .jsonl file
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                total_entries += 1
                
                # Ground truth processing: CLEAN the raw label first
                true_label_raw = entry['ground_truth'].upper().strip()
                true_label_cleaned = normalize_web_attack_label(true_label_raw) 
                
                # Prediction processing
                pred_label_raw = entry['prediction'].upper().strip()
                
                # 1. Determine OVERALL binary labels
                is_true_malicious = (true_label_cleaned != BENIGN_LABEL)
                true_binary = MALICIOUS_LABEL if is_true_malicious else BENIGN_LABEL
                
                # Normalize predicted label to ensure it's strictly BENIGN or MALICIOUS for binary comparison
                if pred_label_raw == MALICIOUS_LABEL:
                    pred_binary = MALICIOUS_LABEL
                else:
                    # Any other predicted label (BENIGN, UNKNOWN, etc.) is treated as BENIGN for the overall matrix
                    pred_binary = BENIGN_LABEL

                true_labels_binary.append(true_binary)
                predicted_labels_binary.append(pred_binary)

                # 2. Determine Per-Attack Metrics (Only if flow is actually malicious)
                if is_true_malicious:
                    specific_attack_type = true_label_cleaned # Use the CLEANED label
                    
                    if specific_attack_type in per_attack_results:
                        
                        # Case 1: Model predicted it as MALICIOUS (True Positive for this attack class)
                        if pred_binary == MALICIOUS_LABEL:
                            per_attack_results[specific_attack_type]['TP'] += 1
                        
                        # Case 2: Model predicted it as BENIGN (False Negative for this attack class)
                        elif pred_binary == BENIGN_LABEL:
                            per_attack_results[specific_attack_type]['FN'] += 1
                            
                    
        
    except FileNotFoundError:
        print(f"Error: Output file not found at {jsonl_file}. Please run main.py first.")
        return

    if total_entries == 0:
        print("No entries found in the output file.")
        return

    # 3. Calculate the Overall Binary Confusion Matrix
    cm_binary = confusion_matrix(true_labels_binary, predicted_labels_binary, labels=LABELS_TO_USE)
    TN, FP, FN, TP = cm_binary.ravel()
    
    # 4. Calculate Overall Binary Metrics (average='binary' is now safe because both arrays are strictly binary)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        true_labels_binary, predicted_labels_binary, 
        average='binary', pos_label=MALICIOUS_LABEL, zero_division=0, labels=LABELS_TO_USE
    )
    accuracy = accuracy_score(true_labels_binary, predicted_labels_binary)

    # 5. Output Summary and Detailed JSON
    
    # --- Print Overall Summary ---
    print("\n" + "="*70)
    print("      OVERALL BINARY PERFORMANCE (MALICIOUS vs. BENIGN)")
    print("="*70)
    print(f"Total Flows Evaluated: {total_entries}")
    
    print("\n--- Confusion Matrix (True vs. Predicted) ---")
    print(f"| {'':<15} | {'Predicted BENIGN':<20} | {'Predicted MALICIOUS':<20} |")
    print("-" * 62)
    print(f"| {'True BENIGN':<15} | {TN:<20} | {FP:<20} | (False Alarms)")
    print(f"| {'True MALICIOUS':<15} | {FN:<20} | {TP:<20} | (Missed Attacks)")
    print("-" * 62)
    
    print("\n--- Overall Model Performance Metrics ---")
    print(f"1. Accuracy: {accuracy:.4f}")
    print(f"2. Precision (Malicious): {precision:.4f}")
    print(f"3. Recall (Malicious - Critical Metric): {recall:.4f}")
    print(f"4. F1-Score: {f1_score:.4f}")
    print("="*70)

    # 6. Prepare Detailed Per-Attack JSON Output
    detailed_metrics_json = {}
    
    # Calculate FP and TN for each attack based on the overall count of Benign flows
    # FIX: Cast to standard Python int to prevent JSON TypeError
    total_benign_flows_int = int(TN + FP) # All flows that were actually BENIGN
    
    print("\n--- PER-ATTACK BREAKDOWN (Debugging Focus) ---")
    
    for attack, counts in per_attack_results.items():
        TP_count = counts['TP']
        FN_count = counts['FN']
        
        # Calculate performance specifically for the attack class
        
        # Assume all Benign flows were TN for this specific attack type, 
        # as a non-DDoS flow is technically a True Negative for the DDoS class.
        TN_count = total_benign_flows_int 
        
        total_attack_samples = TP_count + FN_count
        
        # Calculate specific Recall: how many of the actual attacks did we catch?
        attack_recall = TP_count / total_attack_samples if total_attack_samples > 0 else 0
        
        
        detailed_metrics_json[attack] = {
            "Total_Samples": total_attack_samples,
            "True_Positive_TP": TP_count,
            "False_Negative_FN": FN_count,
            "Total_Benign_Samples_TN_FP_Base": TN_count, # Use the int variable
            "Recall_Detection_Rate": attack_recall
        }
        
        # Print key debugging metric
        if total_attack_samples > 0:
            print(f"[{attack:<20}] Total: {total_attack_samples:<8} | TP: {TP_count:<8} | FN: {FN_count:<8} | Recall: {attack_recall:.4f}")
        else:
            print(f"[{attack:<20}] No samples of this attack type were found in the results.")


    # Save the detailed per-attack breakdown to a separate JSON file
    detailed_filename = "per_attack_metrics.json"
    with open(detailed_filename, 'w') as f:
        json.dump(detailed_metrics_json, f, indent=4)
        
    print(f"\nDetailed per-attack metrics saved to {detailed_filename}")
    print("="*70)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model results from a .jsonl file.")
    parser.add_argument(
        '--file', 
        type=str, 
        default='classification_output.jsonl', 
        help="Path to the JSON Lines file containing predictions and ground truth."
    )
    args = parser.parse_args()
    calculate_evaluation_metrics(args.file)