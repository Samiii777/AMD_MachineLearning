#!/bin/bash

cd ~/src

# Clone the repository
git clone https://github.com/Samiii777/AMD_MachineLearning.git

# Change directory to AMD_MachineLearning/linux/
cd AMD_MachineLearning/linux/

# Run the install script
sh ./install.sh


cd ~/src

# Clone the repository
git clone https://github.com/farshadghodsian/llava-amd-radeon-demo

# Change directory to the cloned repository
cd llava-amd-radeon-demo

# Ensure conda is available in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create a conda environment named llava by cloning base
conda create --name llava --clone base

# Activate the llava environment
conda activate llava

# Upgrade pip (optional, but recommended)
pip install --upgrade pip

# Install required packages
pip install accelerate fastchat gradio transformers==4.37.2

echo "Setup complete. The llava environment is now active and ready to use."
