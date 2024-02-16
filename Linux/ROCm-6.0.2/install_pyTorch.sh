#!/bin/bash

# Install Pytorch
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl

pip3 install --force-reinstall torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl

