import onnx
import onnxruntime as ort
import torch
import numpy as np
from onnxconverter_common import auto_mixed_precision

# Define the shape of the input tensor for ResNet50
input_shape = (1, 3, 224, 224)

# Create a sample input tensor in FP16
test_input = np.random.rand(*input_shape).astype(np.float32)

# Convert the test input into a PyTorch tensor
test_data = torch.tensor(test_input)

# Contruct dictionary for input tensor
input_dict = {'input.1': test_data.numpy()}

# Load the model
model = onnx.load("resnet50.onnx")

# Convert the model
model_mixed = auto_mixed_precision.auto_convert_mixed_precision(model, input_dict, rtol=0.03, atol=0.003, keep_io_types=True)
onnx.save(model_mixed, "resnet50_mixed.onnx")
