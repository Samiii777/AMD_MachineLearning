#!/bin/bash

current_dir=$(dirname "$0")

# Check if the script is running with root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script with root privileges (sudo)."
    exit 1
fi

# Install pip3
apt install -y python3-pip 
#Install Jupyter-lab
snap install jupyter
apt install -y jupyter-core

pip3 install jupyterlab

export PATH="$HOME/.local/bin:$PATH"
