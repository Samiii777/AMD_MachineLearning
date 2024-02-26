import torch

# Check if GPU is available through pytorch
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("ERROR: GPU is not available.")