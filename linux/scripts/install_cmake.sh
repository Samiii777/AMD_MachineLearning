#!/bin/bash

# Default CMake version
DEFAULT_CMAKE_VERSION="3.28.3"

# Use user-provided CMake version if provided, otherwise use default
CMAKE_VERSION=${1:-$DEFAULT_CMAKE_VERSION}

# URL for CMake download
URL_BASE="https://github.com/Kitware/CMake/releases/download/"

sudo apt update

# Check if libssl-dev is installed

if [ "$(dpkg -l | awk '/'libssl-dev'/ {print }' | wc -l)" -ge 1 ]; then
    echo "libssl-dev is already installed."
else
    echo "Installing libssl-dev..."
    sudo apt install libssl-dev -y
fi

# Remove the existing CMake installation
sudo apt autoremove cmake
sudo apt clean

# Download and install the specified CMake version
wget "${URL_BASE}v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz"
tar xvf cmake-${CMAKE_VERSION}.tar.gz
cd cmake-${CMAKE_VERSION}
./bootstrap && make && sudo make install

# Remove downloaded file and extracted folder
cd ..
rm -f cmake-${CMAKE_VERSION}.tar.gz
sudo rm -rf cmake-${CMAKE_VERSION}


# Echo installed CMake version
installed_cmake_version=$(cmake --version | head -n 1 | awk '{print $3}')
echo "CMake version $installed_cmake_version has been installed."
