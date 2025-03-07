#!/bin/bash

# Check if the script is running with root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script with root privileges (sudo)."
    exit 1
fi

# Download AMD Driver from http://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/
wget -r -np -nH --cut-dirs=4 -P /tmp/ https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/jammy/amdgpu-install_6.3.60304-1_all.deb

# Install AMD Driver Package
apt-get install -y /tmp/amdgpu-install_6.3.60304-1_all.deb

# Install AMD Driver for Graphics and ROCm
## replace graphics with workstation if you are using a Workstation Graphich Card
amdgpu-install -y --usecase=graphics,rocm


