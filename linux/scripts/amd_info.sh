#!/bin/bash

# Extract ROCm version installed on the system
rocm_version=$(apt-cache show rocm-libs | grep Version | grep -oP 'Version: \K\d+\.\d+\.\d+')

# Check if the version is non-empty
if [ -n "$rocm_version" ]; then
    echo "ROCm version installed: $rocm_version"
else
    echo "Unable to determine ROCm version."
fi
