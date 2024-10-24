import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import urllib
import onnxruntime
import argparse
import time
import onnx
import onnxmltools


def get_default_provider():
    available_providers = onnxruntime.get_available_providers()
    
    if 'MIGraphXExecutionProvider' in available_providers:
        return 'migraphx' 
    elif 'ROCMExecutionProvider' in available_providers:
        return 'rocm'
    else:
        return 'cpu'

def onnxInference(params):
    download_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    input_image = Image.open("dog.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if params.EP:
        EP = params.EP.lower()
    else:
        EP = get_default_provider()

    model_path = "resnet50_fp16.onnx" if params.fp16 else "resnet50.onnx"

    print(f"GPU Availability: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    else:
        print("GPU unavailable. Defaulting to CPU.")

    if params.fp16:
        input_batch = input_batch.to(torch.float16)

    if EP == 'migraphx':
        session = onnxruntime.InferenceSession(model_path, providers=['MIGraphXExecutionProvider'])
    elif EP == 'rocm':
        session = onnxruntime.InferenceSession(model_path, providers=['ROCMExecutionProvider'])
    else:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Warm-up run
    if torch.cuda.is_available():
        input_batch = input_batch.cpu()
    ort_outputs = session.run(None, {'input.1': input_batch.numpy()})[0]

    latency = []
    torch.cuda.synchronize()
    start = time.time()
    if torch.cuda.is_available():
        input_batch = input_batch.cpu()
    ort_outputs = session.run(None, {'input.1': input_batch.numpy()})[0]
    torch.cuda.synchronize()
    end = time.time()
    latency.append(end - start)

    output = ort_outputs.flatten()
    output = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))  # softmax
    top5_catid = np.argsort(-output)[:5]

    for i in range(len(top5_catid)):
        print(categories[top5_catid[i]], output[top5_catid[i]])

    print(f"ONNX Runtime {'FP16' if params.fp16 else 'FP32'} with {EP} Inference time = {format(sum(latency) * 1000 / len(latency), '.2f')} ms")

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EP", type=str, required=False, help="Execution Provider: MIGraphX, ROCm, CPU")
    parser.add_argument("--fp16", action='store_true', help="Use FP16 precision for inference")
    args = parser.parse_args()
    onnxInference(args)

if __name__ == '__main__':
    main()
