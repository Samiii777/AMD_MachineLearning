import torch
import torchvision
import datetime

# Try to import triton, but make it optional
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

LOG_FILE = "log.txt"

# Function to log messages
def log(level, message):
    # Print to console
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")
    # Also write to log file
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}\n")


# Check if GPU is available through PyTorch
if torch.cuda.is_available():
    log("PYTORCH_TEST", "GPU is available.")
    log("PYTORCH_TEST", f"Installed version of PyTorch: {torch.__version__}")
    log("PYTORCH_TEST", f"Installed version of TorchVision: {torchvision.__version__}")
    
    if TRITON_AVAILABLE:
        log("PYTORCH_TEST", f"Installed version of Triton: {triton.__version__}")
    else:
        log("PYTORCH_TEST", "Triton is not installed.")
    
    # Additional GPU information
    log("PYTORCH_TEST", f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log("PYTORCH_TEST", f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test basic tensor operations on GPU
    try:
        device = torch.device("cuda:0")
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.mm(x, y)
        log("PYTORCH_TEST", "Basic GPU tensor operations successful!")
    except Exception as e:
        log("PYTORCH_TEST", f"GPU tensor operation failed: {e}")
        
else:
    log("PYTORCH_TEST", "GPU is not available. It is advised to restart your system if you haven't done so yet and try again after the restart.")
    log("PYTORCH_TEST", "Also make sure you have added the user to the render and video group.")
    
    # Test CPU operations
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        log("PYTORCH_TEST", "CPU tensor operations successful!")
    except Exception as e:
        log("PYTORCH_TEST", f"CPU tensor operation failed: {e}")