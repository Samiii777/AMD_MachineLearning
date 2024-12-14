#!/bin/bash

# Install git if not already installed
sudo apt-get install -y git

# Clone the tinygrad repository
git clone https://github.com/tinygrad/tinygrad.git

# Change to the tinygrad directory
cd tinygrad

# Install tinygrad
python3 -m pip install -e .

# Check if tinygrad is installed
if python3 -c "import tinygrad" &> /dev/null; then
    echo "tinygrad has been installed successfully!"
else
    echo "Failed to install tinygrad. Please check for errors."
fi
