import argparse
import time
import torch
import torchvision.models as models

def warm_up_model(model, input_tensor, device):
    # Warm-up the model
    with torch.no_grad():
        model(input_tensor.to(device))
        torch.cuda.synchronize()

def run_benchmarking(params):
    device = torch.device("cuda")
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)

    warm_up_model(model, input_tensor, device)

    input_tensor = input_tensor.to(device)

    num_trials = params.iterations
    num_runs = 3
    total_average_time_ms = 0.0

    for run in range(num_runs):
        total_time = 0.0

        for _ in range(num_trials):
            start_time = time.time()
            with torch.no_grad():
                model(input_tensor)
                torch.cuda.synchronize()
            total_time += (time.time() - start_time) * 1000  # Convert to milliseconds

        average_time_ms = total_time / num_trials
        print(f"Average inference time for ResNet50 on {device} (Run {run+1}): {average_time_ms:.5f} ms")
        total_average_time_ms += average_time_ms

        if run < num_runs - 1:
            print("Sleeping for 2 seconds before next run...")
            time.sleep(2)

    final_average_time_ms = total_average_time_ms / num_runs
    print(f"Final Average inference time for ResNet50 on {device}: {final_average_time_ms:.5f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, required=False, default=100, help="Iterations")
    args = parser.parse_args()

    run_benchmarking(args)

if __name__ == '__main__':
    main()
