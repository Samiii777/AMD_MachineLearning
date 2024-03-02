#!/bin/bash

###TODO 1 click install for everything including driver, pytorch and onnx
## Everything Required for AMD ML Stuff

# Update and Install Linux to the Latest Version
sudo usermod -aG render $(whoami)
sudo usermod -aG video $(whoami)

sudo apt update
sudo dist-upgrade -y

# Install pip3
sudo apt install python3-pip -y
pip3 install wheel setuptools

# Install Libstdc++-12-dev that is required for pytorch
sudo apt install libstdc++-12-dev -y
sudo apt install libclblast-dev -y

sudo sh rocm/rocm-6.0.2/install_amd_driver_with_rocm_on_ubuntu.sh
sudo sh rocm/rocm-6.0.2/install_pyTorch.sh
sudo sh rocm/rocm-6.0.2/install_OnnxRT.sh

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
