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

wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl

pip3 install --force-reinstall torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl

# Clean up downloaded files
rm torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl
