import argparse
import time
import torch
import os
from diffusers import StableDiffusionPipeline
import random

# Default models and precisions
DEFAULT_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1"
]
DEFAULT_PRECISIONS = ["fp16", "fp32"]

def run_benchmarking(params):
    device_ids = params.device_ids
    ngpus = len(device_ids.split(',')) if params.device_ids else torch.cuda.device_count()
    iterations = params.iterations
    batch_size = 1  # Since we're dealing with a single inference
    seed = random.randint(0, 2**32 - 1)

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

    for model_id in models:
        for precision_str in precisions:
            precision = torch.float32 if precision_str == 'fp32' else torch.float16
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=precision)
            pipe = pipe.to("cuda")

            # Set the random seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Perform a dry run (skip timing for the first iteration)
            _ = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

            tm = time.time()
            for i in range(iterations - 2):  # Exclude the first two iterations from timing
                image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            torch.cuda.synchronize()
            tm2 = time.time()

            time_per_inference = (tm2 - tm) / (iterations - 2)  # Exclude the first two iterations from timing

            # Save the last generated image
            model_name = model_id.split('/')[-1]
            image_filename = f"{model_name}_{precision_str}_seed{seed}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)

            print("OK: finished running benchmark..")
            print("--------------------SUMMARY--------------------------")
            print("Benchmark for Stable Diffusion Pipeline")
            print("Model: {}".format(model_id))
            print("Precision: {}".format(precision_str))
            print("Num devices: {}".format(ngpus))
            print("Mini batch size [img] : {}".format(batch_size))
            print("Time per inference : {:.4f}".format(time_per_inference))
            print("Throughput [img/sec] : {:.4f}".format(batch_size / time_per_inference))
            print("Image saved as: {}".format(image_path))
            print("----------------------------------------------------")
            print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=False, help="Comma-separated list of model IDs for the Stable Diffusion Pipeline")
    parser.add_argument("--precisions", type=str, required=False, help="Comma-separated list of precisions (fp16 or fp32)")
    parser.add_argument("--iterations", type=int, required=False, default=5, help="Iterations")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on.")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed for image generation")
    args = parser.parse_args()

    run_benchmarking(args)

if __name__ == '__main__':
    main()
