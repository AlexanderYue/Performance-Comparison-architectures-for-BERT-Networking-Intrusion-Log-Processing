## 🛡️ BERT-based Network Intrusion Detection Benchmarking Project

This project evaluates the **inference performance** of a **BERT-based Network Intrusion Detection (NID) classifier** across diverse hardware architectures, including **NVIDIA CUDA**, **AMD CPUs/GPUs**, and **Apple Silicon (MPS)**.

-----

## 🛠️ Prerequisites

  * **Python 3.8 or higher.**
  * **Required Libraries:** Install all necessary dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

-----

## 📂 Project Files Overview

| File | Description | Target Use Case |
| :--- | :--- | :--- |
| **`main.py`** | **Main benchmarking script.** Targets NVIDIA CUDA or AMD/CPU fallback. | Windows/Linux PCs |
| **`main_mac.py`** | **Adapted benchmarking script** for Apple Silicon (M1/M3) using the **MPS** backend. | macOS |
| **`evaluate.py`** | Analyzes benchmark outputs to generate **model quality metrics** (Recall, Precision, Confusion Matrix). | Post-benchmark analysis |
| **`filter.py`** | Utility to categorize and filter different attack types from the dataset. | Data preparation |
| **`merge_csv.py`** | Script to combine multiple dataset CSV files into a single file. | Data preparation |
| **`splitter.py`** | Creates a **subsampled dataset** ($\approx 1/6$ size) for faster testing on resource-constrained hardware (e.g., CPUs). | Data preparation |
| **`benchmark_colab.ipynb`** | Jupyter notebook for testing on Google Colab (e.g., NVIDIA A100). *Requires proper import/extraction of `archive.zip`.* | Cloud GPU Testing |

-----

## ⚙️ Setup Instructions

### 1\. Importing Model from Hugging-Face

1.  Go to https://huggingface.co/yashika0998/IoT-23-BERT-Network-Logs-Classification
2.  Navigate to "Files and Versions"
3.  Download the following files:
    -config.json
    -pytorch_model.bin
    -ONNX-model-Network-Logs-Classification.onnx
5.  Create a new directory named `local_model_files` in the project root.
6.  Move the downloaded content into the created directory.


### 2\. Prepare Test Logs

1.  Create a new directory named `test_logs` in the project root.
2.  Download **CIC-IDS2017 CSV files**
3.  Copy `archive.zip` into the `test_logs` directory.
4.  **Extract its contents** into `test_logs`. This provides the **CIC-IDS2017 CSV files** used for benchmarking.
5.  **Verification:** In `main.py` or `main_mac.py`, confirm that the `TEST_LOGS_DIR` variable correctly points to this path (e.g., `"./test_logs"`).

### 3\. Optional: Dataset Preparation

  * **Merging:** Use `python merge_csv.py` if your raw CSV files need to be combined.
  * **Subsampling (Recommended for CPU):** For hardware where full-dataset inference is too slow ($\approx 2.83$ million flows), run `python splitter.py` to generate a smaller, subsampled dataset.

-----

## 🚀 Running the Benchmark

Choose the appropriate script (`main.py` or `main_mac.py`) and use the `--device` argument to select your target hardware. Run all commands from the project root.

### Command Examples

| Target Hardware | Device Flag | Command to Run |
| :--- | :--- | :--- |
| **NVIDIA CUDA** (RTX 2080, A100, etc.) | `cuda` or `nvidia` | `python main.py --device cuda` |
| **Apple Silicon** (M1/M3) | `apple` or `mps` | `python main_mac.py --device mps` |
| **AMD/CPU Fallback** | `cpu` or `radeon` | `python main.py --device cpu` |
| **Auto-Detect** (Default) | `auto` (or omit) | `python main.py` |

### Optional Arguments

  * **Classification Threshold:** Override the default threshold (0.85) using `--threshold <value>`.
      * *Example:* `python main.py --device cuda --threshold 0.90`

### Output

The script performs the following steps:

1.  Loads and preprocesses the dataset.
2.  Runs model inference.
3.  Outputs raw classification results to **`classification_output.jsonl`**.
4.  Outputs raw performance metrics to **`benchmark_report.json`**.

### 📊 Model Quality Metrics

After running the benchmark, execute **`evaluate.py`** to process the output file and compute metrics like Recall and Precision:

```bash
python evaluate.py --file classification_output.jsonl
```

-----

## 💡 Notes

  * **Batch Size:** Benchmarks use a default batch size of **256** for optimal GPU utilization. This can be adjusted within the scripts if necessary.
  * **Google Colab:** When using `benchmark_colab.ipynb`, you must upload and extract `archive.zip` directly within the notebook session environment.
  * **Troubleshooting:** If you encounter model or tokenizer loading errors, ensure all dependencies are correctly installed (`requirements.txt`) and that local model files are accessible at the specified path.

Would you like me to draft a *requirements.txt* file based on the context of this project?
