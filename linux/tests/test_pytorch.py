import torch
import torchvision
import triton

# Check if GPU is available through pytorch
if torch.cuda.is_available():
    print("GPU is available.")
    print("Installed version of PyTorch:")
    print(torch.__version__)
    print("Installed version of TorchVision:")
    print(torchvision.__version__)
    print("Installed version of Triton:")
    print(triton.__version__)
else:
    print("ERROR: GPU is not available. It is advised to restart your system if you haven't done so yet and try again after the restart.")
    print("Also make sure you have added the user to the render and video group.")