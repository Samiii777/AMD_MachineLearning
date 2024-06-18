from torchvision import models, datasets, transforms
import torch
from PIL import Image
import numpy as np
import os
import urllib
import onnxruntime
import argparse
import time

def exportToOnnx():

  resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')

  # Check if imagenet_classes.txt exists, otherwise download it
  if not os.path.exists("imagenet_classes.txt"):
      print("Downloading imagenet_classes.txt...")
      url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
      urllib.request.urlretrieve(url, "imagenet_classes.txt")
      print("imagenet_classes.txt downloaded.")

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
                    input_names = ['input'],      # the model's input names
                    output_names = ['output'])    # the model's output names

def get_default_provider():
    available_providers = onnxruntime.get_available_providers()
    
    if 'MIGraphXExecutionProvider' in available_providers:
        return 'migraphx'
    elif 'ROCMExecutionProvider' in available_providers:
        return 'rocm'
    else:
        return 'cpu'


def onnxInference(params):
  # Inference with ONNX Runtime
  # Check if dog.jpg exists, otherwise download it
  if not os.path.exists("dog.jpg"):
      print("Downloading dog.jpg...")
      url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
      urllib.request.urlretrieve(url, "dog.jpg")
      print("dog.jpg downloaded.")

  with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

  input_image = Image.open("dog.jpg")
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
  
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

  def softmax(x):
      """Compute softmax values for each sets of scores in x."""
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum()
  
  latency = []

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
  input_arr = input_batch.cpu().detach().numpy()

  
  ort_outputs = session_fp32.run(None, {'input': input_arr})[0]
  torch.cuda.synchronize()
  start = time.time()
  ort_outputs = session_fp32.run(None, {'input': input_arr})[0]
  torch.cuda.synchronize()
  latency.append(time.time() - start)
  
  output = ort_outputs.flatten()
  output = softmax(output)  # this is optional
  top5_catid = np.argsort(-output)[:5]
  for catid in top5_catid:
    print(categories[catid], output[catid])

  print("ONNX Runtime with {} Inference time = {} ms".format(EP, format(sum(latency) * 1000 / len(latency), '.2f')))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EP", type=str, required=False, help="Execution Provider: ROCm, MIGraphX, CPU")
    args = parser.parse_args()

    exportToOnnx()
    onnxInference(args)

if __name__ == '__main__':
    main()
