import torch
import torchvision.models as models

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet50 model with default weights and move it to the GPU
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()

# Create a dummy input tensor and move it to the GPU
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export the model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx"
)

print("ResNet50 model has been converted to ONNX format and saved as 'resnet50.onnx'")
