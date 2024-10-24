import torch
import torchvision.models as models
import onnx
from onnxconverter_common import float16

def convert_to_fp16(model_path, output_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    
    # Convert the model to FP16
    model_fp16 = float16.convert_float_to_float16(model)
    
    # Save the FP16 model
    onnx.save(model_fp16, output_path)
    print(f"FP16 model saved as {output_path}")

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Convert the ONNX model to FP16
convert_to_fp16("resnet50.onnx", "resnet50_fp16.onnx")

print("Conversion process completed.")
