from torchvision import models, transforms
import torch
from PIL import Image
import numpy as np
import os
import urllib
import onnxruntime
import argparse
import time
import onnx
from onnxconverter_common import float16

# Export the pytorch ResNet50 model to ONNX format
def exportToOnnx():
    try:
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        x = torch.randn(1, 3, 224, 224)
    
        torch.onnx.export(resnet50,                     # model being run
                          x,                            # model input (or a tuple for multiple inputs)
                          "resnet50.onnx")              # where to save the model (can be a file or file-like object)

        print("ResNet50 model successfully exported to ONNX format.")
    except Exception as e:
        print(f"Error exporting ResNet50 model to ONNX format: {e}")

def convert_model_to_fp16(onnx_model_path, fp16_model_path):
    model = onnx.load(onnx_model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, fp16_model_path)
    print(f"Model converted to FP16 and saved as {fp16_model_path}")

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

    model_path = "resnet50_fp16.onnx" if params.fp16 else "resnet50.onnx"

    if EP == 'rocm':
        session = onnxruntime.InferenceSession(model_path, providers=['ROCMExecutionProvider'])
    elif EP == 'migraphx':
        session = onnxruntime.InferenceSession(model_path, providers=['MIGraphXExecutionProvider'])
    elif EP == 'cpu':
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        raise ValueError(f"Invalid execution provider: {EP}")

    latency = []
    input_arr = input_batch.cpu().detach().numpy()
    
    # Convert input to float16 if using FP16 model
    if params.fp16:
        input_arr = input_arr.astype(np.float16)

    # Warm-up run
    session.run(None, {'input': input_arr})
    
    # Timed inference run
    torch.cuda.synchronize()
    start = time.time()
    ort_outputs = session.run(None, {'input': input_arr})[0]
    torch.cuda.synchronize()
    
    latency.append(time.time() - start)

    output = ort_outputs.flatten()
    output = softmax(output)
    
    top5_catid = np.argsort(-output)[:5]
    
    for catid in top5_catid:
        print(categories[catid], output[catid])

    print(f"ONNX Runtime {'FP16' if params.fp16 else 'FP32'} with {EP} Inference time = {format(sum(latency) * 1000 / len(latency), '.2f')} ms")

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
    parser.add_argument("--fp16", action='store_true', help="Use FP16 precision for inference")
    args = parser.parse_args()

    if args.fp16:
        if not os.path.exists("resnet50_fp16.onnx"):
            if not os.path.exists("resnet50.onnx"):
                print("Resnet50.onnx model has NOT been found, Exporting Resnet50 from Pytorch to Onnx format...!")
                exportToOnnx()
            print("Converting ResNet50 ONNX model to FP16...")
            convert_model_to_fp16("resnet50.onnx", "resnet50_fp16.onnx")
            print("Conversion complete.")
        else:
            print("Resnet50_fp16.onnx model has been found, running the inference")
    else:
        if not os.path.exists("resnet50.onnx"):
            print("Resnet50.onnx model has NOT been found, Exporting Resnet50 from Pytorch to Onnx format...!")
            exportToOnnx()
        else:
            print("Resnet50.onnx model has been found, running the inference")

    onnxInference(args)

if __name__ == '__main__':
    main()


