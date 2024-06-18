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

wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/torch-2.1.2%2Brocm6.1.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/torchvision-0.16.1%2Brocm6.1.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/pytorch_triton_rocm-2.1.0%2Brocm6.1.3.4d510c3a44-cp310-cp310-linux_x86_64.whl

pip3 uninstall torch torchvision pytorch-triton-rocm
pip3 install torch-2.1.2+rocm6.1.3-cp310-cp310-linux_x86_64.whl torchvision-0.16.1+rocm6.1.3-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-2.1.0+rocm6.1.3.4d510c3a44-cp310-cp310-linux_x86_64.whl

# Clean up downloaded files
rm torch*.whl pytorch*.whl
