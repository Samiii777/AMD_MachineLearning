import argparse
import time
import torch
import os
import random
import numpy as np
import subprocess
from datetime import datetime
from diffusers import StableDiffusionPipeline

# Default models and precisions
DEFAULT_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1"
]
DEFAULT_PRECISIONS = ["fp16", "fp32"]
BATCH_SIZES = [1]

def get_rocm_version():
    try:
        return subprocess.check_output(["cat", "/opt/rocm/.info/version"], universal_newlines=True).strip()
    except:
        return "unknown"

def run_benchmarking(params):
    device_ids = params.device_ids
    ngpus = len(device_ids.split(',')) if params.device_ids else torch.cuda.device_count()
    iterations = params.iterations

    if device_ids:
        assert ngpus == len(device_ids.split(','))
        torch.cuda.set_device("cuda:%d" % int(device_ids.split(',')[0]))
    else:
        torch.cuda.set_device("cuda:0")

    models = params.models.split(',') if params.models else DEFAULT_MODELS
    precisions = params.precisions.split(',') if params.precisions else DEFAULT_PRECISIONS

    prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."

    # Create output directory if it doesn't exist
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Create benchmark results file
    rocm_version = get_rocm_version()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_file = f"benchmark_results_{rocm_version}_{timestamp}.md"

    with open(benchmark_file, "w") as f:
        f.write(f"# Stable Diffusion Benchmark Results\n\n")
        f.write(f"ROCm Version: {rocm_version}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for model_id in models:
        for precision_str in precisions:
            precision = torch.float32 if precision_str == 'fp32' else torch.float16
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=precision)
            pipe = pipe.to("cuda")

            with open(benchmark_file, "a") as f:
                f.write(f"# Model: {model_id}\n\n")
                f.write(f"## Precision: {precision_str}\n\n")
                f.write("| Batch Size | Time per Batch (s) | Time per Image (s) | Throughput (img/sec) |\n")
                f.write("|------------|--------------------|--------------------|----------------------|\n")

            for batch_size in BATCH_SIZES:
                # Generate random seeds for this batch
                seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
                
                # Perform a dry run (skip timing for the first iteration)
                _ = pipe([prompt] * batch_size, num_inference_steps=50, guidance_scale=7.5, generator=[torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]).images

                tm = time.time()
                for i in range(iterations - 2):  # Exclude the first two iterations from timing
                    images = pipe([prompt] * batch_size, num_inference_steps=50, guidance_scale=7.5, generator=[torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]).images
                torch.cuda.synchronize()
                tm2 = time.time()

                time_per_batch = (tm2 - tm) / (iterations - 2)  # Exclude the first two iterations from timing
                time_per_image = time_per_batch / batch_size
                throughput = batch_size / time_per_batch

                # Save all generated images from the last batch
                model_name = model_id.split('/')[-1]
                for idx, (image, seed) in enumerate(zip(images, seeds)):
                    image_filename = f"{model_name}_{precision_str}_batch{batch_size}_seed{seed}_img{idx}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)

                # Append results to the benchmark file
                with open(benchmark_file, "a") as f:
                    f.write(f"| {batch_size:10d} | {time_per_batch:18.4f} | {time_per_image:18.4f} | {throughput:20.4f} |\n")

                print(f"Completed benchmark for {model_id}, {precision_str}, batch size {batch_size}")
                print(f"Time per batch: {time_per_batch:.4f}s, Time per image: {time_per_image:.4f}s, Throughput: {throughput:.4f} img/sec")
                print(f"Sample images saved in: {output_dir}")
                print("----------------------------------------------------")
                print()

            # Add a newline after each precision's table
            with open(benchmark_file, "a") as f:
                f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=False, help="Comma-separated list of model IDs for the Stable Diffusion Pipeline")
    parser.add_argument("--precisions", type=str, required=False, help="Comma-separated list of precisions (fp16 or fp32)")
    parser.add_argument("--iterations", type=int, required=False, default=5, help="Iterations")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on.")
    args = parser.parse_args()

    run_benchmarking(args)

if __name__ == '__main__':
    main()
