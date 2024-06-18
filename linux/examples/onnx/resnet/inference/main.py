from torchvision import models, transforms
import torch
from PIL import Image
import numpy as np
import os
import urllib
import onnxruntime
import argparse
import time

def exportToOnnx():
  try:
    resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # Export the model to ONNX
    image_height = 224
    image_width = 224
    x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
    torch_out = resnet50(x)
  
    torch.onnx.export(resnet50,                     # model being run
                      x,                            # model input (or a tuple for multiple inputs)
                      "resnet50.onnx",              # where to save the model (can be a file or file-like object)
                      export_params=True,           # store the trained parameter weights inside the model file
                      opset_version=12,             # the ONNX version to export the model to
                      do_constant_folding=True,     # whether to execute constant folding for optimization
                      input_names=['input'],        # the model's input names
                      output_names=['output'])      # the model's output names
    print("ResNet50 model successfully exported to ONNX format.")
  except Exception as e:
    print(f"Error exporting ResNet50 model to ONNX format: {e}")


def get_default_provider():
    available_providers = onnxruntime.get_available_providers()
    
    if 'MIGraphXExecutionProvider' in available_providers:
        return 'migraphx'
    elif 'ROCMExecutionProvider' in available_providers:
        return 'rocm'
    else:
        return 'cpu'


def onnxInference(params):
    """
    Perform inference using the ONNX Runtime.
    """
    download_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

    categories = load_categories("imagenet_classes.txt")
    input_batch = preprocess_image("dog.jpg")

    if params.EP:
        EP = params.EP.lower()
    else:
        EP = get_default_provider()

    if EP == 'rocm':
        session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['ROCMExecutionProvider'])
    elif EP == 'migraphx':
        session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['MIGraphXExecutionProvider'])
    elif EP == 'cpu':
        session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['CPUExecutionProvider'])
    else:
        raise ValueError(f"Invalid execution provider: {EP}")

    latency = []
    input_arr = input_batch.cpu().detach().numpy()
    ort_outputs = session_fp32.run(None, {'input': input_arr})[0]
    torch.cuda.synchronize()
    start = time.time()
    ort_outputs = session_fp32.run(None, {'input': input_arr})[0]
    torch.cuda.synchronize()
    latency.append(time.time() - start)

    output = ort_outputs.flatten()
    output = softmax(output)
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])

    print(f"ONNX Runtime with {EP} Inference time = {format(sum(latency) * 1000 / len(latency), '.2f')} ms")

def download_file(url, filename):
    """
    Download a file from the given URL and save it with the specified filename.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded.")

def load_categories(filename):
    """
    Load the categories from the given file.
    """
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def preprocess_image(image_path):
    """
    Preprocess the input image for inference.
    """
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EP", type=str, required=False, help="Execution Provider: ROCm, MIGraphX, CPU")
    args = parser.parse_args()
    if not os.path.exists("resnet50.onnx"):
      print("Resnet50.onnx model has NOT been found, Exporting Resnet50 from Pytorch to Onnx format...!")
      exportToOnnx()
    else:
      print("Resnet50.onnx model has been found, running the inference")
    onnxInference(args)

if __name__ == '__main__':
    main()
