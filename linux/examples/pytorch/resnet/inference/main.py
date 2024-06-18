import torch
from torchvision import transforms
from PIL import Image
import urllib
import os
import argparse
import time

def run_inference(params):

  resnet_model = "resnet50"  # Default model
  if params.model:
      resnet_model = params.model

# Check if imagenet_classes.txt exists, otherwise download it
  if not os.path.exists("imagenet_classes.txt"):
      print("Downloading imagenet_classes.txt...")
      url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
      urllib.request.urlretrieve(url, "imagenet_classes.txt")
      print("imagenet_classes.txt downloaded.")
  
  # Check if dog.jpg exists, otherwise download it
  if not os.path.exists("dog.jpg"):
      print("Downloading dog.jpg...")
      url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
      urllib.request.urlretrieve(url, "dog.jpg")
      print("dog.jpg downloaded.")
  
  # Load model
  if (resnet_model == "resnet50"):
    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, weights='ResNet50_Weights.DEFAULT')
  elif (resnet_model == "resnet101"):
    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, weights='ResNet101_Weights.DEFAULT')  
  elif (resnet_model == "resnet18"):
    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, weights='ResNet18_Weights.DEFAULT')  
  
  model.eval()
  
  # Load input image
  input_image = Image.open("dog.jpg")
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
  
  # Move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda:0')
      model.to('cuda:0')

 # warm-up run to load model and data onto gpu first
  with torch.no_grad():
      output = model(input_batch)

 # perform inference and record time
  latency = []
  torch.cuda.synchronize()
  start = time.time()
  with torch.no_grad():
      output = model(input_batch)
  torch.cuda.synchronize()
  end = time.time()
  latency.append(end - start)
   
 # Calculate probabilities and print top predictions
  probabilities = torch.nn.functional.softmax(output[0], dim=0)
  with open("imagenet_classes.txt", "r") as f:
      categories = [s.strip() for s in f.readlines()]
  
  top5_prob, top5_catid = torch.topk(probabilities, 5)
  for i in range(top5_prob.size(0)):
     print(categories[top5_catid[i]], top5_prob[i].item())

  print("PyTorch Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, help="Model ID for the RESNET Pipeline")
    args = parser.parse_args()

    run_inference(args)

if __name__ == '__main__':
    main()
