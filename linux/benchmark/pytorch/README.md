# Run Benchmarks that uses Pytorch

This script benchmarks the performance of the Stable Diffusion Pipeline model using PyTorch.

## Prerequisites

- Python 3.x
- PyTorch
- diffusers package (for Stable Diffusion Pipeline)

## Usage

### Running Stable Diffusion Pipeline Benchmark

```bash
python3 stable_diffusion.py [--model MODEL_ID] [--iterations ITERATIONS] [--precision {fp16,fp32}]
```
###### Arguments

- `[--model]`: (Optional) Model ID for the Stable Diffusion Pipeline. Defaults to "runwayml/stable-diffusion-v1-5" if not specified.
- `[--iterations]`: (Optional) Number of iterations. Defaults to 5 if not specified.
- `[--precision]`: (Optional) Precision for inference. Choose between 'fp16' or 'fp32'. Defaults to 'fp16' if not specified.

### Running RESNET50 Benchmark

```bash
python3 resnet.py [--iterations ITERATIONS]
```

- `[--iterations]`: (Optional) Number of iterations. Defaults to 100 if not specified.
