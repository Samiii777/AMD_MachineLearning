import argparse
import time
import torch
import os
import random
import numpy as np
import subprocess
import csv
from datetime import datetime
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline

# Enhanced configurations for each model, including batch sizes and precisions
MODEL_CONFIGS = {
    "CompVis/stable-diffusion-v1-4": {
        "height": 512,
        "width": 512,
        "steps": 28,
        "guidance_scale": 7.0,
        "batch_sizes": [1, 2,4],
        "precisions": ["fp16", "fp32"]
    },
    "runwayml/stable-diffusion-v1-5": {
        "height": 512,
        "width": 512,
        "steps": 28,
        "guidance_scale": 7.5,
        "batch_sizes": [1, 2, 4],
        "precisions": ["fp16", "fp32"]
    },
    "stabilityai/stable-diffusion-2-1": {
        "height": 768,
        "width": 768,
        "steps": 30,
        "guidance_scale": 7.5,
        "batch_sizes": [1, 2],  # Larger model, smaller batch sizes
        "precisions": ["fp16"]  # fp16 recommended for this model
    },
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "height": 1024,
        "width": 1024,
        "steps": 28,
        "guidance_scale": 7.0,
        "batch_sizes": [1],    # Largest model, smallest batch size
        "precisions": ["fp16"]  # fp16 recommended for this model
    }
}

# Default models for benchmarking
DEFAULT_MODELS = [
    #"stabilityai/stable-diffusion-3-medium-diffusers",
    #"runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4"
    #"stabilityai/stable-diffusion-2-1"
]

# Default global settings if not specified in model configs
DEFAULT_PRECISIONS = ["fp16", "fp32"]
DEFAULT_BATCH_SIZES = [1]

def get_rocm_version():
    try:
        return subprocess.check_output(["cat", "/opt/rocm/.info/version"], universal_newlines=True).strip()
    except:
        return "unknown"

def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            return f"{gpu_name} ({gpu_memory:.2f} GB)"
        else:
            return "No GPU detected"
    except:
        return "Unknown GPU"

def run_benchmarking(params):
    device_ids = params.device_ids
    ngpus = len(device_ids.split(',')) if params.device_ids else torch.cuda.device_count()
    iterations = params.iterations
    use_attention_slicing = params.attention_slicing
    use_cpu_offload = params.cpu_offload
    override_batch_sizes = params.batch_sizes.split(',') if params.batch_sizes else None
    override_precisions = params.precisions.split(',') if params.precisions else None

    if device_ids:
        assert ngpus == len(device_ids.split(','))
        torch.cuda.set_device("cuda:%d" % int(device_ids.split(',')[0]))
    else:
        torch.cuda.set_device("cuda:0")

    models = params.models.split(',') if params.models else DEFAULT_MODELS

    prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
    negative_prompt = ""

    # Create output directory if it doesn't exist
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Create benchmark results file (both CSV and MD formats)
    rocm_version = get_rocm_version()
    gpu_info = get_gpu_info()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    md_file = f"SD_benchmark_results_{rocm_version}_{timestamp}.md"
    csv_file = f"SD_benchmark_results_{rocm_version}_{timestamp}.csv"
    
    # Initialize CSV file with headers
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            'timestamp', 'model_id', 'model_name', 'precision', 'height', 'width', 'steps', 
            'guidance_scale', 'batch_size', 'attention_slicing', 'cpu_offload', 
            'time_per_batch', 'time_per_image', 'throughput', 'rocm_version', 'gpu_info', 'status'
        ])

    # Create markdown file with system info
    with open(md_file, "w") as f:
        f.write(f"# Stable Diffusion Benchmark Results\n\n")
        f.write(f"ROCm Version: {rocm_version}\n")
        f.write(f"GPU: {gpu_info}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Memory Optimizations: Attention Slicing: {use_attention_slicing}, CPU Offload: {use_cpu_offload}\n\n")
        f.write(f"## Models Tested\n")
        for model in models:
            f.write(f"- {model}\n")
        f.write("\n")

    for model_id in models:
        # Get the appropriate configuration for this model or use defaults
        config = MODEL_CONFIGS.get(model_id, {
            "height": 512, 
            "width": 512,
            "steps": 28,
            "guidance_scale": 7.0,
            "batch_sizes": DEFAULT_BATCH_SIZES,
            "precisions": DEFAULT_PRECISIONS
        })
        
        # Use model-specific configs but allow command-line overrides
        batch_sizes = [int(b) for b in override_batch_sizes] if override_batch_sizes else config["batch_sizes"]
        precisions = override_precisions if override_precisions else config["precisions"]
        
        # Get a short model name for display purposes
        model_name = model_id.split('/')[-1]
        
        for precision_str in precisions:
            precision = torch.float32 if precision_str == 'fp32' else torch.float16
            
            # Load the appropriate model
            if "stabilityai/stable-diffusion-3-medium-diffusers" in model_id:
                pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=precision)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=precision)
            
            # Apply memory optimization techniques if requested
            if use_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
                
            if use_attention_slicing:
                pipe.enable_attention_slicing()

            with open(md_file, "a") as f:
                f.write(f"# Model: {model_id}\n\n")
                f.write(f"## Configuration\n")
                f.write(f"- Height: {config['height']}\n")
                f.write(f"- Width: {config['width']}\n")
                f.write(f"- Steps: {config['steps']}\n")
                f.write(f"- Guidance Scale: {config['guidance_scale']}\n")
                f.write(f"- Precision: {precision_str}\n")
                f.write(f"- Batch Sizes: {batch_sizes}\n\n")
                f.write("| Batch Size | Time per Batch (s) | Time per Image (s) | Throughput (img/sec) |\n")
                f.write("|------------|--------------------|--------------------|----------------------|\n")

            for batch_size in batch_sizes:
                # Generate random seeds for this batch
                seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
                
                try:
                    # Perform a dry run (skip timing for the first iteration)
                    _ = pipe(
                        prompt=[prompt] * batch_size,
                        negative_prompt=[negative_prompt] * batch_size,
                        num_inference_steps=config["steps"],
                        height=config["height"],
                        width=config["width"],
                        guidance_scale=config["guidance_scale"],
                        generator=[torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
                    ).images

                    tm = time.time()
                    for i in range(iterations - 2):  # Exclude the first two iterations from timing
                        images = pipe(
                            prompt=[prompt] * batch_size,
                            negative_prompt=[negative_prompt] * batch_size,
                            num_inference_steps=config["steps"],
                            height=config["height"],
                            width=config["width"],
                            guidance_scale=config["guidance_scale"],
                            generator=[torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
                        ).images
                    torch.cuda.synchronize()
                    tm2 = time.time()

                    time_per_batch = (tm2 - tm) / (iterations - 2)  # Exclude the first two iterations from timing
                    time_per_image = time_per_batch / batch_size
                    throughput = batch_size / time_per_batch

                    # Save all generated images from the last batch
                    for idx, (image, seed) in enumerate(zip(images, seeds)):
                        image_filename = f"{model_name}_{precision_str}_batch{batch_size}_seed{seed}_img{idx}.png"
                        image_path = os.path.join(output_dir, image_filename)
                        image.save(image_path)

                    # Append results to the markdown file
                    with open(md_file, "a") as f:
                        f.write(f"| {batch_size:10d} | {time_per_batch:18.4f} | {time_per_image:18.4f} | {throughput:20.4f} |\n")
                    
                    # Write results to CSV file
                    with open(csv_file, 'a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            model_id,
                            model_name,
                            precision_str,
                            config["height"],
                            config["width"],
                            config["steps"],
                            config["guidance_scale"],
                            batch_size,
                            use_attention_slicing,
                            use_cpu_offload,
                            f"{time_per_batch:.4f}",
                            f"{time_per_image:.4f}",
                            f"{throughput:.4f}",
                            rocm_version,
                            gpu_info,
                            "success"
                        ])

                    print(f"Completed benchmark for {model_id}, {precision_str}, batch size {batch_size}")
                    print(f"Time per batch: {time_per_batch:.4f}s, Time per image: {time_per_image:.4f}s, Throughput: {throughput:.4f} img/sec")
                    print(f"Sample images saved in: {output_dir}")
                    print("----------------------------------------------------")
                    print()
                
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error running benchmark for {model_id} with {precision_str} precision and batch size {batch_size}:")
                    print(error_msg)
                    
                    # Log the error in the markdown file
                    with open(md_file, "a") as f:
                        f.write(f"| {batch_size:10d} | ERROR: {error_msg[:100]}... | | |\n")
                    
                    # Log the error in the CSV file
                    with open(csv_file, 'a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            model_id,
                            model_name,
                            precision_str,
                            config["height"],
                            config["width"],
                            config["steps"],
                            config["guidance_scale"],
                            batch_size,
                            use_attention_slicing,
                            use_cpu_offload,
                            "0",
                            "0",
                            "0",
                            rocm_version,
                            gpu_info,
                            f"error: {error_msg[:100]}"
                        ])

            # Add a newline after each precision's table
            with open(md_file, "a") as f:
                f.write("\n")

            # Clean up to free memory
            del pipe
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=False, help="Comma-separated list of model IDs for the Stable Diffusion Pipeline")
    parser.add_argument("--precisions", type=str, required=False, help="Comma-separated list of precisions (fp16 or fp32) to override model defaults")
    parser.add_argument("--batch_sizes", type=str, required=False, help="Comma-separated list of batch sizes to override model defaults")
    parser.add_argument("--iterations", type=int, required=False, default=5, help="Iterations")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on.")
    parser.add_argument("--attention_slicing", action='store_true', help="Enable attention slicing to reduce memory usage")
    parser.add_argument("--cpu_offload", action='store_true', help="Enable CPU offload to reduce memory usage")
    args = parser.parse_args()

    run_benchmarking(args)

if __name__ == '__main__':
    main()