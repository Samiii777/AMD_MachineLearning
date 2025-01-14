import time
from tinygrad.tensor import Tensor
from tinygrad import TinyJit
from tinygrad.device import Device
from tinygrad.nn.state import get_parameters
from extra.models.resnet import ResNet50

def run_benchmarking():
    device = Device.DEFAULT
    print(f"Using device: {device}")

    # Load the actual ResNet50 model
    model = ResNet50()

    input_tensor = Tensor.randn(1, 3, 224, 224)

    @TinyJit
    def run_model(x):
        return model(x)

    # Warm up
    for _ in range(10):
        run_model(input_tensor)

    num_trials = 100
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
        print(f"Average inference time for ResNet50 (Run {run+1}): {average_time_ms:.5f} ms")
        total_average_time_ms += average_time_ms

    print(f"Overall average time: {(total_average_time_ms/num_runs):.5f} ms")

if __name__ == "__main__":
    run_benchmarking()
