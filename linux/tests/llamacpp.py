import os
import argparse
from huggingface_hub import hf_hub_download
import subprocess
from datetime import datetime
import git
import csv

# If llama.cpp folder exists
if os.path.exists("llama.cpp"):
    print("llama.cpp folder found.")
else:
    print("llama.cpp folder not found. Cloning repository.")
    repo = git.Repo.clone_from("https://github.com/ggerganov/llama.cpp.git", "llama.cpp")
    subprocess.run(["make", "clean"], cwd="llama.cpp")
    subprocess.run(["make", "GGML_HIPBLAS=1"], cwd="llama.cpp")

# Define the model directory
MODEL_DIR = "llama.cpp/models"
LLAMACPP_DIR = "llama.cpp"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
rocmVersion = subprocess.check_output(["cat", "/opt/rocm/.info/version"], universal_newlines=True).strip()
BENCHMARK_FILE = f"llamaCpp_benchmark_results_{rocmVersion}_{timestamp}.md"
CSV_BENCHMARK_FILE = f"llamaCpp_benchmark_results_{rocmVersion}_{timestamp}.csv"

# Dictionary of model files and their corresponding repo IDs
MODEL_FILES = {
    "LLAMA2": {
        "Chat": {
            "7B": {
                "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                "files": [
                    "llama-2-7b-chat.Q4_0.gguf",
                    "llama-2-7b-chat.Q4_K_M.gguf",
                    "llama-2-7b-chat.Q4_K_S.gguf",
                    "llama-2-7b-chat.Q8_0.gguf"
                ]
            },
        },
    },
    "LLAMA3": {
        "Instruct": {
            "8B": {
                "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                "files": [
                    "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                    "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                    "Meta-Llama-3-8B-Instruct.Q4_K_S.gguf",
                    "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
                ]
            },
        }
    },
    "LLAMA3.1": {
        "": {
            "7B": {
                "repo_id": "SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
                "files": [
                    "meta-llama-3.1-8b-instruct.Q4_0.gguf",
                    "meta-llama-3.1-8b-instruct.Q4_K_M.gguf",
                    "meta-llama-3.1-8b-instruct.Q4_K_S.gguf",
                    "meta-llama-3.1-8b-instruct.Q8_0.gguf"
                ]
            },
            "70B": {
                 "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
                 "files": [
                     "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
                ]
            },
        }
    }
}

# Function to download a file
def download_file(repo_id, file_name):
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded {file_name}")
    except Exception as e:
        print(f"Failed to download {file_name}: {str(e)}")

# Function to parse benchmark output and save to CSV
def parse_and_save_benchmark(output, csv_file, model_info):
    lines = output.split('\n')
    start_index = next((i for i, line in enumerate(lines) if line.startswith('| model')), -1)
    if start_index == -1:
        print("Benchmark table not found in output")
        return
    
    header = [col.strip() for col in lines[start_index].split('|')[1:-1]]
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # Write header only if file is empty
            writer.writerow(['Model'] + header)
        
        # Write model info and benchmark results
        for line in lines[start_index+2:]:  # Skip the header and separator lines
            if line.startswith('|'):
                row = [col.strip() for col in line.split('|')[1:-1]]
                writer.writerow([model_info] + row)
        
        # Add an empty row for better readability
        writer.writerow([])

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download GGUF model files and run llama-cli")
parser.add_argument("--model-type", choices=list(MODEL_FILES.keys()), help="Type of model to download (LLAMA2, LLAMA3, Gemma)")
parser.add_argument("--model-variant", choices=["Chat", "Instruct", ""], help="Variant of model to download (optional)")
parser.add_argument("--model-size", help="Size of the model to download (optional)")
parser.add_argument("--prompt", default="Hi you how are you", help="Prompt for llama-cli")
parser.add_argument("--n", type=int, default=50, help="Number of tokens to generate")
parser.add_argument("--echo", action="store_true", help="Echo prompt")
parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
parser.add_argument("--threads", type=int, default=4, help="Number of threads")
args = parser.parse_args()

# Determine which model type, variant, and sizes to process
if args.model_type:
    model_types = [args.model_type]
else:
    model_types = MODEL_FILES.keys()

if args.model_variant:
    model_variants = [args.model_variant]
else:
    model_variants = ["Chat", "Instruct", ""]

if args.model_size:
    model_sizes = [args.model_size]
else:
    model_sizes = ["7B", "8B", "13B", "70B"]

# Process each selected model type, variant, and size
for model_type in model_types:
    if model_type in MODEL_FILES:
        for variant in model_variants:
            if variant in MODEL_FILES[model_type]:
                for size in model_sizes:
                    if size in MODEL_FILES[model_type][variant]:
                        print(f"\nProcessing {model_type} {variant} {size} model files:")
                        repo_id = MODEL_FILES[model_type][variant][size]["repo_id"]
                        files = MODEL_FILES[model_type][variant][size]["files"]

                        # Check each file and download if necessary
                        for file in files:
                            file_path = os.path.join(MODEL_DIR, file)
                            if os.path.exists(file_path):
                                print(f"The file {file} already exists in {MODEL_DIR}")
                            else:
                                print(f"The file {file} does not exist in {MODEL_DIR}. Downloading...")
                                download_file(repo_id, file)

                        # Run llama-cli for each downloaded model
                        for file in files:
                            file_path = os.path.join("models", file)
                            print(f"\nRunning llama-cli with model: {file}")
                            command = f"./llama-cli -m {file_path} -p '{args.prompt}' -n {args.n} -e -ngl {args.top_k} -t {args.threads}"
                            subprocess.run(command, shell=True, cwd=LLAMACPP_DIR)

                        # Run llama-bench for each downloaded model
                        for file in files:
                            file_path = os.path.join("models", file)
                            print(f"\nRunning llama-bench for {model_type} {variant} {size} model: {file}")
                            bench_command = f"./llama-bench -m {file_path}"
                            
                            # Run the benchmark and capture the output
                            result = subprocess.run(bench_command, shell=True, cwd=LLAMACPP_DIR, capture_output=True, text=True)
                            
                            # Write the full output to the markdown file
                            with open(BENCHMARK_FILE, "a") as f:
                                f.write(f"\n## Benchmark results for {model_type} {variant} {size} model: {file}\n\n")
                                f.write(result.stdout)
                                f.write(result.stderr)
                            
                            # Parse the output and save to CSV
                            model_info = f"{model_type} {variant} {size} - {file}"
                            parse_and_save_benchmark(result.stdout, CSV_BENCHMARK_FILE, model_info)

                            # Print a separator for better readability
                            print("-" * 50)

print("\nAll requested model files have been processed and llama-cli run for each.")
print(f"Benchmark results saved to: {BENCHMARK_FILE}")
print(f"Benchmark results also saved to CSV: {CSV_BENCHMARK_FILE}")
