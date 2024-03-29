#!/bin/bash

## TODO - Create a logging
# Default ROCm version
rocm_version="6.0.2"

# Check if a version argument is provided
if [ $# -eq 1 ]; then
  rocm_version=$1
fi

# Update and Install Linux to the Latest Version
sudo usermod -aG render $(whoami)
sudo usermod -aG video $(whoami)

sudo apt update
sudo dist-upgrade -y

# Install pip3
sudo apt install python3-pip -y
pip3 install wheel setuptools

## TODO - Create a package checker function
# Install Libstdc++-12-dev that is required for pytorch
sudo apt install libstdc++-12-dev -y
sudo apt install libclblast-dev -y

# Install ROCm with the specified version
sudo sh "rocm/rocm-${rocm_version}/install_amd_driver_with_rocm_on_ubuntu.sh"
sudo sh "rocm/rocm-${rocm_version}/install_pyTorch.sh"
sudo sh "rocm/rocm-${rocm_version}/install_OnnxRT.sh"

pip3 install -r requirements.txt
# Prompt the user to reboot or not
read -p "Do you want to reboot the system now? (y/n): " choice
case "$choice" in
  y|Y )
    shutdown -r now ;;
  n|N )
    echo "Reboot skipped. You may need to restart the system later for changes to take effect." ;;
  * )
    echo "Invalid choice. Reboot skipped. You may need to restart the system later for changes to take effect." ;;
esac
