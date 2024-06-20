import os
import argparse
from huggingface_hub import hf_hub_download
import subprocess
from datetime import datetime
import git


# If llama.cpp folder exists
if os.path.exists("llama.cpp"):
    print("llama.cpp folder found.")
else:
    print("llama.cpp folder not found. Cloning repository.")
    repo = git.Repo.clone_from("https://github.com/ggerganov/llama.cpp.git", "llama.cpp")
    subprocess.run(["make", "clean"], cwd="llama.cpp")
    subprocess.run(["make", "LLAMA_HIPBLAS=1"], cwd="llama.cpp")

# Define the model directory
MODEL_DIR = "llama.cpp/models"
LLAMACPP_DIR = "llama.cpp"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
rocmVersion = subprocess.check_output(["cat", "/opt/rocm/.info/version"], universal_newlines=True).strip()
BENCHMARK_FILE = f"benchmark_results_{rocmVersion}_{timestamp}.md"

# Dictionary of model files and their corresponding repo IDs
MODEL_FILES = {
    "LLAMA2": {
        "Chat": {
            "7B": {
                "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                "files": [
                    "llama-2-chat-7b.Q2_K.gguf",
                    # "llama-2-chat-7b.Q3_K_L.gguf",
                    # "llama-2-chat-7b.Q3_K_M.gguf",
                    # "llama-2-chat-7b.Q3_K_S.gguf",
                    # "llama-2-chat-7b.Q4_0.gguf",
                    # "llama-2-chat-7b.Q4_K_M.gguf",
                    # "llama-2-chat-7b.Q4_K_S.gguf",
                    # "llama-2-chat-7b.Q5_0.gguf",
                    # "llama-2-chat-7b.Q5_K_M.gguf",
                    # "llama-2-chat-7b.Q5_K_S.gguf",
                    # "llama-2-chat-7b.Q6_K.gguf",
                    # "llama-2-chat-7b.Q8_0.gguf"
                ]
            },
            "13B": {
                "repo_id": "QuantFactory/Meta-Llama-2-Chat-13B-GGUF",
                "files": [
                    # "llama-2-chat-13b.Q2_K.gguf",
                    # "llama-2-chat-13b.Q3_K_L.gguf",
                    # "llama-2-chat-13b.Q3_K_M.gguf",
                    # "llama-2-chat-13b.Q3_K_S.gguf",
                    # "llama-2-chat-13b.Q4_0.gguf",
                    # "llama-2-chat-13b.Q4_K_M.gguf",
                    # "llama-2-chat-13b.Q4_K_S.gguf",
                    # "llama-2-chat-13b.Q5_0.gguf",
                    # "llama-2-chat-13b.Q5_K_M.gguf",
                    # "llama-2-chat-13b.Q5_K_S.gguf",
                    # "llama-2-chat-13b.Q6_K.gguf",
                    # "llama-2-chat-13b.Q8_0.gguf"
                ]
            }
        },
        "Instruct": {
            "7B": {
                "repo_id": "TheBloke/Llama-2-Instruct-7B-GGUF",
                "files": [
                    # "llama-2-instruct-7b.Q2_K.gguf",
                    # "llama-2-instruct-7b.Q3_K_L.gguf",
                    # "llama-2-instruct-7b.Q3_K_M.gguf",
                    # "llama-2-instruct-7b.Q3_K_S.gguf",
                    # "llama-2-instruct-7b.Q4_0.gguf",
                    # "llama-2-instruct-7b.Q4_K_M.gguf",
                    # "llama-2-instruct-7b.Q4_K_S.gguf",
                    # "llama-2-instruct-7b.Q5_0.gguf",
                    # "llama-2-instruct-7b.Q5_K_M.gguf",
                    # "llama-2-instruct-7b.Q5_K_S.gguf",
                    # "llama-2-instruct-7b.Q6_K.gguf",
                    # "llama-2-instruct-7b.Q8_0.gguf"
                ]
            },
            "13B": {
                "repo_id": "QuantFactory/Meta-Llama-2-Instruct-13B-GGUF",
                "files": [
                    # "llama-2-instruct-13b.Q2_K.gguf",
                    # "llama-2-instruct-13b.Q3_K_L.gguf",
                    # "llama-2-instruct-13b.Q3_K_M.gguf",
                    # "llama-2-instruct-13b.Q3_K_S.gguf",
                    # "llama-2-instruct-13b.Q4_0.gguf",
                    # "llama-2-instruct-13b.Q4_K_M.gguf",
                    # "llama-2-instruct-13b.Q4_K_S.gguf",
                    # "llama-2-instruct-13b.Q5_0.gguf",
                    # "llama-2-instruct-13b.Q5_K_M.gguf",
                    # "llama-2-instruct-13b.Q5_K_S.gguf",
                    # "llama-2-instruct-13b.Q6_K.gguf",
                    # "llama-2-instruct-13b.Q8_0.gguf"
                ]
            }
        }
    },
    "LLAMA3": {
        "Chat": {
            "8B": {
                "repo_id": "QuantFactory/Meta-Llama-3-Chat-8B-GGUF",
                "files": [
                    # "llama-3-chat-8b.Q2_K.gguf",
                    # "llama-3-chat-8b.Q3_K_L.gguf",
                    # "llama-3-chat-8b.Q3_K_M.gguf",
                    # "llama-3-chat-8b.Q3_K_S.gguf",
                    # "llama-3-chat-8b.Q4_0.gguf",
                    # "llama-3-chat-8b.Q4_1.gguf",
                    # "llama-3-chat-8b.Q4_K_M.gguf",
                    # "llama-3-chat-8b.Q4_K_S.gguf",
                    # "llama-3-chat-8b.Q5_0.gguf",
                    # "llama-3-chat-8b.Q5_1.gguf",
                    # "llama-3-chat-8b.Q5_K_M.gguf",
                    # "llama-3-chat-8b.Q5_K_S.gguf",
                    # "llama-3-chat-8b.Q6_K.gguf",
                    # "llama-3-chat-8b.Q8_0.gguf"
                ]
            },
            "13B": {
                "repo_id": "QuantFactory/Meta-Llama-3-Chat-13B-GGUF",
                "files": [
                    # "llama-3-chat-13b.Q2_K.gguf",
                    # "llama-3-chat-13b.Q3_K_L.gguf",
                    # "llama-3-chat-13b.Q3_K_M.gguf",
                    # "llama-3-chat-13b.Q3_K_S.gguf",
                    # "llama-3-chat-13b.Q4_0.gguf",
                    # "llama-3-chat-13b.Q4_1.gguf",
                    # "llama-3-chat-13b.Q4_K_M.gguf",
                    # "llama-3-chat-13b.Q4_K_S.gguf",
                    # "llama-3-chat-13b.Q5_0.gguf",
                    # "llama-3-chat-13b.Q5_1.gguf",
                    # "llama-3-chat-13b.Q5_K_M.gguf",
                    # "llama-3-chat-13b.Q5_K_S.gguf",
                    # "llama-3-chat-13b.Q6_K.gguf",
                    # "llama-3-chat-13b.Q8_0.gguf"
                ]
            }
        },
        "Instruct": {
            "8B": {
                "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                "files": [
                    "Meta-Llama-3-8B-Instruct.Q2_K.gguf",
                    "Meta-Llama-3-8B-Instruct.Q3_K_L.gguf",
                    "Meta-Llama-3-8B-Instruct.Q3_K_M.gguf",
                    "Meta-Llama-3-8B-Instruct.Q3_K_S.gguf",
                    "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                    "Meta-Llama-3-8B-Instruct.Q4_1.gguf",
                    "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                    "Meta-Llama-3-8B-Instruct.Q4_K_S.gguf",
                    "Meta-Llama-3-8B-Instruct.Q5_0.gguf",
                    "Meta-Llama-3-8B-Instruct.Q5_1.gguf",
                    "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
                    "Meta-Llama-3-8B-Instruct.Q5_K_S.gguf",
                    "Meta-Llama-3-8B-Instruct.Q6_K.gguf",
                    "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
                ]
            },
            "13B": {
                "repo_id": "QuantFactory/Meta-Llama-3-Instruct-13B-GGUF",
                "files": [
                    # "llama-3-instruct-13b.Q2_K.gguf",
                    # "llama-3-instruct-13b.Q3_K_L.gguf",
                    # "llama-3-instruct-13b.Q3_K_M.gguf",
                    # "llama-3-instruct-13b.Q3_K_S.gguf",
                    # "llama-3-instruct-13b.Q4_0.gguf",
                    # "llama-3-instruct-13b.Q4_1.gguf",
                    # "llama-3-instruct-13b.Q4_K_M.gguf",
                    # "llama-3-instruct-13b.Q4_K_S.gguf",
                    # "llama-3-instruct-13b.Q5_0.gguf",
                    # "llama-3-instruct-13b.Q5_1.gguf",
                    # "llama-3-instruct-13b.Q5_K_M.gguf",
                    # "llama-3-instruct-13b.Q5_K_S.gguf",
                    # "llama-3-instruct-13b.Q6_K.gguf",
                    # "llama-3-instruct-13b.Q8_0.gguf"
                ]
            }
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


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download GGUF model files and run llama-cli")
parser.add_argument("--model-type", choices=list(MODEL_FILES.keys()), help="Type of model to download (LLAMA2 or LLAMA3)")
parser.add_argument("--model-variant", choices=["Chat", "Instruct"], help="Variant of model to download (optional)")
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
    model_variants = ["Chat", "Instruct"]

if args.model_size:
    model_sizes = [args.model_size]
else:
    model_sizes = ["7B", "8B", "13B"]

# Process each selected model type, variant, and size
for model_type in model_types:
    for variant in model_variants:
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
                    print(f"\nRunning llama-bench with model: {file}")
                    bench_command = f"./llama-bench -m {file_path}"
                    with open(BENCHMARK_FILE, "a") as f:
                        subprocess.run(bench_command, shell=True, cwd=LLAMACPP_DIR, stdout=f, stderr=subprocess.DEVNULL)

print("\nAll requested model files have been processed and llama-cli run for each.")
print(f"Benchmark results saved to: {BENCHMARK_FILE}")
