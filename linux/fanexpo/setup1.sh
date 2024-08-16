#!/bin/bash

sudo apt update -y
sudo apt dist-upgrade -y
sudo apt install git -y
sudo apt install openssh-server -y

mkdir -p ~/src

# Create directory for Miniconda
mkdir -p ~/miniconda3

# Download the specified Miniconda version
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash

echo "Miniconda installation complete. Please restart your terminal or run 'source ~/.bashrc' to use Miniconda."
