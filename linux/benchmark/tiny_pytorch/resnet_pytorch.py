import argparse
import time
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn

def run_benchmarking():
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load the actual ResNet50 model
    model = models.resnet50(weights=None).to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    num_trials = 100
    num_runs = 3
    total_average_time_ms = 0.0

    for run in range(num_runs):
        total_time = 0.0
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                _ = model(input_tensor)

            torch.cuda.synchronize()
            total_time += (time.time() - start_time) * 1000  # Convert to milliseconds

        average_time_ms = total_time / num_trials
        print(f"Average inference time for ResNet50 (Run {run+1}): {average_time_ms:.5f} ms")
        total_average_time_ms += average_time_ms

    print(f"Overall average time: {(total_average_time_ms/num_runs):.5f} ms")

if __name__ == "__main__":
    run_benchmarking()
