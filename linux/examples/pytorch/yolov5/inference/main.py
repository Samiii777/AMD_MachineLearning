import torch
import cv2
import numpy as np
import urllib
import os
import argparse
import time

def run_inference(params):
    # Load YOLOv5 model
    yolo_model = "yolov5l.pt"  # You can change to yolov5m.pt, yolov5l.pt, or yolov5x.pt for larger models

    
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    # Prepare input image
    image_path = "dog.jpg"
    download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", image_path)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Warm-up run
    with torch.no_grad():
        _ = model(img)

    # Perform inference and record time
    latency = []
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        results = model(img)
    
    torch.cuda.synchronize()
    latency.append(time.time() - start)

    # Print results
    results.print()  # Print results to console

    # Show results on the image
    results.show()

    print(f"YOLOv5 Inference Time = {format(sum(latency) * 1000 / len(latency), '.2f')} ms\n")

def download_file(url, filename):
    """
    Download a file from the given URL and save it with the specified filename.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, help="Model ID for the YOLOv5 Pipeline")
    args = parser.parse_args()

    run_inference(args)

if __name__ == '__main__':
    main()
