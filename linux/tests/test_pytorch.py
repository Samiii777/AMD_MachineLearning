import torch

# Check if GPU is available through pytorch
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("ERROR: GPU is not available. It is advised to restart your system if you haven't done so yet and try again after the restart.")