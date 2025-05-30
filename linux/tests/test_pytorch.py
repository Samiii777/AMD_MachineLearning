import torch
import torchvision
import triton
import datetime

LOG_FILE = "log.txt"

# Function to log messages
def log(level, message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}\n")

# Check if GPU is available through PyTorch
if torch.cuda.is_available():
    log("PYTORCH_TEST", "GPU is available.")
    log("PYTORCH_TEST", f"Installed version of PyTorch: {torch.__version__}")
    log("PYTORCH_TEST", f"Installed version of TorchVision: {torchvision.__version__}")
    log("PYTORCH_TEST", f"Installed version of Triton: {triton.__version__}")
else:
    log("PYTORCH_TEST", "GPU is not available. It is advised to restart your system if you haven't done so yet and try again after the restart.")
    log("PYTORCH_TEST", "Also make sure you have added the user to the render and video group.")