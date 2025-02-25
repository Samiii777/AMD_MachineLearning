#!/bin/bash

# Install Onnx Runtime
sudo apt install migraphx -y
sudo apt install half -y
pip3 uninstall onnxruntime-rocm onnxruntime -y
pip3 install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.2/

