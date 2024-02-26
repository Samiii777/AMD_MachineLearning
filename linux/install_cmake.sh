#!/bin/bash

#Change the cmake version, with your desired one
URL_BASE="https://github.com/Kitware/CMake/releases/download/"
CMAKE_VERSION="3.28.3"

sudo apt-get update
sudo apt-get install libssl-dev

sudo apt autoremove cmake
sudo apt clean
wget "${URL_BASE}v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz"
tar xvf cmake-${CMAKE_VERSION}.tar.gz
cd cmake-${CMAKE_VERSION}
./bootstrap && make && sudo make install

# Remove downloaded file and extracted folder
cd ..
rm -f cmake-${CMAKE_VERSION}.tar.gz
sudo rm -rf cmake-${CMAKE_VERSION}
