import argparse
import time
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
from tinygrad import TinyJit
from tinygrad.device import Device


def resnet50():
    # This is a simplified ResNet50 model for demonstration purposes
    return Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

def run_benchmarking(params):
    device = Device.DEFAULT
    print(device)
    model = resnet50()

    input_tensor = Tensor.randn(1, 3, 224, 224)

    @TinyJit
    def run_model(x):
        return model(x)

    num_trials = params.iterations
    num_runs = 3
    total_average_time_ms = 0.0

    for run in range(num_runs):
        total_time = 0.0

        for _ in range(num_trials):
            Device.default.synchronize()
            start_time = time.time()
            run_model(input_tensor)
            Device.default.synchronize()
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
