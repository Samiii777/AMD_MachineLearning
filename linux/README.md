## Overview

This guide includes an automatic installation method for setting up the AMD driver with ROCm, along with installation instructions for popular machine learning frameworks such as PyTorch, TensorFlow, and ONNX Runtime.

## Installation

### Automatic Installation

To automatically install the AMD driver with ROCm, PyTorch, TensorFlow, and ONNXRuntime, simply run the following command:

```bash
bash ./install.sh [ROCM_VERSION]
```

- **Arguments**:
  - `[ROCM_VERSION]`: Specify the ROCm version to install. Defaults to `6.3.4` if not provided.

### Configuration

The `install.sh` script uses a configuration file named `rocm_config.yml` to determine the correct URLs for downloading the necessary packages for the specified ROCm version. The configuration file should include the following structure:

```yaml
versions:
  "6.3.4":
    rocm:
      version: "6.3.4"
      driver_url: "<driver_url>"
    pytorch:
      wheel_urls:
        torch: "<torch_wheel_url>"
        torchvision: "<torchvision_wheel_url>"
        triton: "<triton_wheel_url>"
        torchaudio: "<torchaudio_wheel_url>"
    tensorflow:
      wheel_url: "<tensorflow_wheel_url>"
    onnxruntime:
      repo_url: "<onnxruntime_repo_url>"

