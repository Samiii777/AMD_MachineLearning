#!/bin/bash

# Install Onnx Runtime
sudo apt install migraphx -y
sudo apt install half -y
pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl

