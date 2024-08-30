#!/bin/bash

current_dir=$(dirname "$0")

# Check if the script is running with root privileges

# Install pip3
sudo apt install -y python3-pip 
#Install Jupyter-lab
sudo snap install jupyter
sudo apt install -y jupyter-core

pip3 install jupyterlab

export PATH="$HOME/.local/bin:$PATH"
