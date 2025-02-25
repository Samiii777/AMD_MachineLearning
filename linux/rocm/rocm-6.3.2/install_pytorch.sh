#!/bin/bash

package_name="libstdc++-12-dev"

# Check if the package is installed
if [ "$(dpkg -l | awk '/'$package_name'/ {print }' | wc -l)" -ge 1 ]; then
    echo "$package_name is already installed."
else
    # If the package is not installed, install it
    echo "$package_name is not installed. Installing..."
    sudo apt update
    sudo apt install $package_name -y
    echo "$package_name has been successfully installed."
fi

wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.2/torch-2.4.0%2Brocm6.3.2-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.2/torchvision-0.19.0%2Brocm6.3.2-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.2/pytorch_triton_rocm-3.0.0%2Brocm6.3.2.75cc27c26a-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.2/torchaudio-2.4.0%2Brocm6.3.2-cp310-cp310-linux_x86_64.whl

pip3 uninstall torch torchvision pytorch-triton-rocm torchaudio -y
pip3 install torch-2.4.0+rocm6.3.2-cp310-cp310-linux_x86_64.whl torchvision-0.19.0+rocm6.3.2-cp310-cp310-linux_x86_64.whl torchaudio-2.4.0+rocm6.3.2-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-3.0.0+rocm6.3.2.75cc27c26a-cp310-cp310-linux_x86_64.whl


# Clean up downloaded files
rm torch*.whl pytorch*.whl
