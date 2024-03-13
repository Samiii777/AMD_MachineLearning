import argparse
import sys
import time
import torch
from diffusers import StableDiffusionPipeline

def run_benchmarking(params):
    device_ids = params.device_ids
    ngpus = len(device_ids.split(',')) if params.device_ids else torch.cuda.device_count()
    iterations = params.iterations
    batch_size = 1  # Since we're dealing with a single inference

    if device_ids:
        assert ngpus == len(device_ids)
        torch.cuda.set_device("cuda:%d" % device_ids[0])
    else:
        torch.cuda.set_device("cuda:0")

    model_id = "runwayml/stable-diffusion-v1-5"  # Default model ID
    if params.model:
        model_id = params.model

    precision = torch.float32 if params.precision == 'fp32' else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=precision)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"

    tm = time.time()
    for i in range(iterations):
        image = pipe(prompt).images[0]
    torch.cuda.synchronize()
    tm2 = time.time()

    time_per_inference = (tm2 - tm) / iterations

    print("OK: finished running benchmark..")
    print("--------------------SUMMARY--------------------------")
    print("Benchmark for Stable Diffusion Pipeline")
    print("Model: {}".format(model_id))
    print("Num devices: {}".format(ngpus))
    print("Mini batch size [img] : {}".format(batch_size))
    print("Time per inference : {}".format(time_per_inference))
    print("Throughput [img/sec] : {}".format(batch_size / time_per_inference))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, help="Model ID for the Stable Diffusion Pipeline")
    parser.add_argument("--iterations", type=int, required=False, default=20, help="Iterations")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on.")
    parser.add_argument("--precision", type=str, required=False, default='fp16', choices=['fp16', 'fp32'], help="Precision for inference (choose between fp16 or fp32)")
    args = parser.parse_args()

    run_benchmarking(args)

if __name__ == '__main__':
    main()
