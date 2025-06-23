# AMD Machine Learning on Linux

A comprehensive guide and automation toolkit for setting up AMD GPUs with ROCm for machine learning workloads on Linux systems, including WSL support.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Advanced Installation](#advanced-installation)
  - [Virtual Environment Setup](#virtual-environment-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Examples](#examples)
  - [Benchmarks](#benchmarks)
  - [Testing](#testing)
- [Supported Versions](#supported-versions)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This toolkit provides automated installation and configuration for AMD machine learning environments on Linux, featuring:

- **Automated ROCm Installation**: One-command setup for AMD drivers and ROCm
- **ML Framework Support**: PyTorch, TensorFlow, and ONNX Runtime with ROCm acceleration
- **WSL Compatibility**: Full support for Windows Subsystem for Linux
- **Version Management**: Support for multiple ROCm versions (6.3.4, 6.4.1)
- **Ubuntu Support**: Compatible with Ubuntu 22.04 and 24.04
- **Example Projects**: Ready-to-run examples for various ML workloads
- **Benchmarking Tools**: Performance testing utilities for your setup

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 22.04 or 24.04 (native Linux or WSL)
- **AMD GPU**: Compatible with ROCm (see [AMD ROCm compatibility](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html))
- **Python**: 3.10 (Ubuntu 22.04) or 3.12 (Ubuntu 24.04)
- **Internet Connection**: Required for downloading packages
- **Disk Space**: At least 10GB free space for complete installation

### Required Tools

The installation script will automatically install these dependencies:
- `curl`, `wget`
- `yq` (YAML processor)
- `python3-pip`, `python3-venv`
- Build tools (`build-essential`, `cmake`)

## Installation

### Quick Start

For a complete automated installation with default settings:

```bash
# Clone the repository
git clone https://github.com/Samiii777/AMD_MachineLearning.git
cd AMD_MachineLearning/linux

# Run the installation script
bash ./install.sh
```

This will install:
- AMD GPU drivers
- ROCm 6.4.1 (default version)
- PyTorch with ROCm support
- TensorFlow with ROCm support
- ONNX Runtime with ROCm support

### Advanced Installation

#### Specify ROCm Version

```bash
# Install specific ROCm version
bash ./install.sh 6.3.4
```

#### Install with Virtual Environment

```bash
# Create and use virtual environment
bash ./install.sh --venv

# Or specify custom venv path
bash ./install.sh --venv --venv-path ./my_ml_env
```

#### WSL-Specific Installation

The script automatically detects WSL and applies appropriate settings:

```bash
# Same command works for WSL
bash ./install.sh
```

**Note**: For WSL, ensure you have WSL 2 with GPU support enabled.

### Installation Options

| Option | Description | Example |
|--------|-------------|---------|
| `[ROCM_VERSION]` | Specify ROCm version | `bash ./install.sh 6.3.4` |
| `--venv` | Use virtual environment | `bash ./install.sh --venv` |
| `--venv-path PATH` | Custom venv location | `bash ./install.sh --venv --venv-path ./myenv` |

## Configuration

### ROCm Configuration File

The installation uses `rocm_config.yml` to manage version-specific URLs and settings:

```yaml
metadata:
  default_version: "6.4.1"

versions:
  "6.4.1":
    ubuntu:
      "22.04":
        rocm:
          driver_url: "https://repo.radeon.com/amdgpu-install/..."
        pytorch:
          wheel_urls:
            torch: "https://repo.radeon.com/rocm/manylinux/..."
            torchvision: "https://repo.radeon.com/rocm/manylinux/..."
        # ... additional frameworks
```

### Supported Combinations

| ROCm Version | Ubuntu 22.04 | Ubuntu 24.04 | Python Version |
|--------------|---------------|---------------|----------------|
| 6.4.1        | ✅            | ✅            | 3.10 / 3.12    |
| 6.3.4        | ✅            | ❌            | 3.10           |

## Usage

### Verify Installation

After installation, verify your setup:

```bash
# Check ROCm installation
rocm-smi

# Test PyTorch with ROCm
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Check GPU device name
python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

### Examples

The repository includes practical examples for various ML frameworks:

#### PyTorch Examples

```bash
# ResNet training
cd examples/pytorch/resnet/training
python3 main.py

# ResNet inference
cd examples/pytorch/resnet/inference
python3 main.py

# Multi-GPU training
cd examples/pytorch/resnet/training/mgpu
python3 main.py

# BERT training
cd examples/pytorch/bert/training
python3 main.py

# 3D ResNet
cd examples/pytorch/3dresnet
python3 main.py

# YOLOv5 inference
cd examples/pytorch/yolov5/inference
python3 main.py
```

#### ONNX Runtime Examples

```bash
# ResNet inference with ONNX
cd examples/onnx/resnet/inference
python3 main.py

# Convert to FP16
python3 convert_fp16.py

# Mixed precision conversion
python3 convert_mixed.py
```

#### Other Tools

```bash
# Ollama integration
cd examples/ollama
python3 ollama_tool.py

# CrewAI example
cd examples/crewai
python3 main.py
```

### Benchmarks

Performance benchmarking tools are available to test your setup:

```bash
# PyTorch benchmarks
cd benchmark/pytorch
python3 resnet.py          # ResNet benchmark
python3 llama3.py          # LLaMA-3 benchmark
python3 stable_diffusion.py # Stable Diffusion benchmark

# Ollama benchmarks
cd benchmark/ollama
python3 benchmark.py

# TinyPyTorch benchmarks
cd benchmark/tiny_pytorch
python3 resnet_pytorch.py
python3 resnet_tiny.py
```

### Testing

Run comprehensive tests to validate your installation:

```bash
cd tests

# Test individual frameworks
python3 test_pytorch.py
python3 test_tensorflow.py
python3 test_onnxruntime.py

# Run specific model tests
python3 stable_diffusion.py
python3 llamacpp.py
```

## Supported Versions

### ROCm Versions
- **6.4.1** (Default) - Latest stable version
- **6.3.4** - Previous stable version

### Framework Versions
- **PyTorch**: 2.6.0 (ROCm 6.4.1) / 2.4.0 (ROCm 6.3.4)
- **TensorFlow**: 2.18.1 (ROCm 6.4.1) / 2.17.0 (ROCm 6.3.4)
- **ONNX Runtime**: Latest compatible version per ROCm release

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Check if GPU is visible
lspci | grep -i amd

# Check ROCm status
rocm-smi

# Verify driver installation
dkms status
```

#### 2. PyTorch Not Using GPU

```bash
# Check CUDA availability in PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# Check ROCm environment
echo $ROCM_PATH
```

#### 3. Permission Issues

```bash
# Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Logout and login again
```

#### 4. Ubuntu 24.04 Path Issues

For Ubuntu 24.04, ensure `~/.local/bin` is in your PATH:

```bash
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
source ~/.bashrc
```

#### 5. WSL-Specific Issues

- Ensure WSL 2 is installed with GPU support enabled
- Check Windows GPU drivers are updated
- Verify WSL GPU access: `nvidia-smi` or `rocm-smi`

### Log Files

Installation logs are saved to `log.txt` in the installation directory. Check this file for detailed error information.

### Getting Help

1. Check the [installation log](#log-files) first
2. Verify your system meets the [prerequisites](#prerequisites)
3. Run the verification commands in the [usage section](#verify-installation)
4. Open an issue on GitHub with your log file and system information

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Additional ROCm version support
- New example implementations
- Bug fixes and improvements
- Documentation updates

### Adding New ROCm Versions

To add support for a new ROCm version, update the `rocm_config.yml` file with the appropriate URLs and version information.

---

**Note**: This installation may require a system reboot to fully activate the AMD drivers. The script will notify you if a reboot is necessary.
