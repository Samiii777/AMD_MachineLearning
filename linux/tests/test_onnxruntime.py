import onnxruntime as ort
import datetime

LOG_FILE = "log.txt"

# Function to log messages
def log(level, message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}\n")

# Check ONNX Runtime version and providers
log("INFO", f"Onnxruntime version: {ort.__version__}")

providers = ort.get_available_providers()
log("INFO", f"Available providers: {', '.join(providers)}")

if 'ROCMExecutionProvider' in providers and 'MIGraphXExecutionProvider' in providers:
    log("INFO", "Onnx runtime is installed successfully with both ROCm and MIGraphX providers.")
    print("Onnx runtime is installed successfully with both ROCm and MIGraphX providers.")
else:
    missing_providers = []
    if 'ROCMExecutionProvider' not in providers:
        missing_providers.append('ROCMExecutionProvider')
    if 'MIGraphXExecutionProvider' not in providers:
        missing_providers.append('MIGraphXExecutionProvider')
    
    if missing_providers:
        error_message = f"ERROR: Onnx runtime is not installed with the following provider(s): {', '.join(missing_providers)}"
        log("ERROR", error_message)
        print(error_message)
    else:
        error_message = "ERROR: Onnx runtime is not installed with either ROCm or MIGraphX provider."
        log("ERROR", error_message)
        print(error_message)