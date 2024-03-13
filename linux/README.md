## Overview

This guide includes both automatic and manual installation methods for setting up the AMD driver with ROCm, along with installation instructions for popular machine learning frameworks such as PyTorch and OnnxRuntime.

## Installation

### Automatic Installation

To automatically install the AMD driver with ROCm, PyTorch and ONNXRuntime simply run the following command:

```bash
sh ./install.sh [ROCM_VERSION]
```
###### Arguments

- `[ROCM_VERSION]`: Specify the ROCm version to install. If not provided, the default version `6.0.2` will be used.


### Manual Installation

1. First we need to update our Kernel Version to the latest one
```
sudo apt update
sudo apt dist-upgrade -y
```
2. Run below commands manually in terminal and reboot the system
```
sudo usermod -aG render $LOGNAME
sudo usermod -aG video $LOGNAME
```
3. Then Simply run below bash file from the corresponding desired ROCm version folder
```
cd ROCm-6.0.2
sudo sh ./install_amd_driver_with_rocm_on_ubuntu.sh
```
4. To install Pytorch
```
sh ./install_pyTorch.sh
```
5. To install OnnxRuntime (will install MIGraphX as well)
```
sh ./install_OnnxRT.sh
```
