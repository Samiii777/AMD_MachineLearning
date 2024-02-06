#!/bin/bash

# Check if the script is running with root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script with root privileges (sudo)."
    exit 1
fi


apt update
apt install libjpeg-dev python3-dev python3-pip
pip3 install wheel setuptools

# https://pytorch.org/get-started/locally/
# Install Pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0/
