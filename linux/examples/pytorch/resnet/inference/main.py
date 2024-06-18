import torch
from torchvision import transforms
from PIL import Image
import urllib
import os
import argparse
import time

def run_inference(params):

  model_weights = {
    'resnet18': "ResNet18_Weights.DEFAULT",
    'resnet34': "ResNet34_Weights.DEFAULT",
    'resnet50': "ResNet50_Weights.DEFAULT",
    'resnet101': "ResNet101_Weights.DEFAULT",
    # Add more models as needed
  }
  resnet_model = "resnet50" if not params.model else params.model.lower()

  download_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
  download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

  categories = load_categories("imagenet_classes.txt")
  print(f"{resnet_model.title()}_Weights.DEFAULT")
  model = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, weights=model_weights[resnet_model])
  model.eval()
  
  # Load input image
  input_batch = preprocess_image("dog.jpg")
  
  # Move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

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
  latency.append(time.time() - start)

   
 # Calculate probabilities and print top predictions
  probabilities = torch.nn.functional.softmax(output[0], dim=0)
  top5_prob, top5_catid = torch.topk(probabilities, 5)
  for i in range(top5_prob.size(0)):
     print(categories[top5_catid[i]], top5_prob[i].item())

  print("PyTorch Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))

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
    parser.add_argument("--model", type=str, required=False, help="Model ID for the RESNET Pipeline")
    args = parser.parse_args()

    run_inference(args)

if __name__ == '__main__':
    main()
