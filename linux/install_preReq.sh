#!/bin/bash

###TODO 1 click install for everything including driver, pytorch and onnx
## Everything Required for AMD ML Stuff

# Update and Install Linux to the Latest Version
sudo apt update
sudo dist-upgrade -y

# Install pip3
sudo apt install python3-pip -y
pip3 install wheel setuptools

# Install Libstdc++-12-dev that is required for pytorch
sudo apt install libstdc++-12-dev -y



## Extra Stuff that is not needed for everyone

# Install SSH Server
sudo apt install openssh-server -y

# Install net-tools
sudo apt install net-tools -y

#Install Jupyter-lab
sudo snap install jupyter
sudo apt install jupyter-core
pip3 install jupyterlab
export PATH="$HOME/.local/bin:$PATH"

# Install Git
sudo apt install git -y
sudo apt install git-lfs -y

# Install LightDM without prompts
export DEBIAN_FRONTEND=noninteractive

sudo apt-get install lightdm -y

unset DEBIAN_FRONTEND

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
