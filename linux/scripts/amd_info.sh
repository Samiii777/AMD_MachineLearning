#!/bin/bash

# Function to extract GPU version
getGfxVersion() {
    local gfxVersion=$(rocminfo | awk '/Agent 2/,/^  Name:/ {if ($1 == "Name:") print $2}')

    # Check and print GPU version
    if [ -n "$gfxVersion" ]; then
        echo "gfxVersion: $gfxVersion"
    else
        echo "Unable to determine GPU version."
    fi
}

# Function to extract Marketing Name
getMarketingName() {
    local marketingName=$(rocminfo | awk '/Agent 2/,/^  Marketing Name:/ {if ($1 == "Marketing" && $2 == "Name:") {$1=""; print $0}}' | sed 's/^[ \t]*Name:[ \t]*//')
    # Check and print Marketing Name
    if [ -n "$marketingName" ]; then
        echo "Marketing Name: $marketingName"
    else
        echo "Unable to determine Marketing Name."
    fi
}

# Function to extract ROCm version
getRocmVersion() {
    local rocmVersion=$(apt-cache show rocm-libs | grep Version | grep -oP 'Version: \K\d+\.\d+\.\d+')

    # Check and print ROCm version
    if [ -n "$rocmVersion" ]; then
        echo "ROCm version installed: $rocmVersion"
    else
        echo "Unable to determine ROCm version."
    fi
}

# Function to check and print all information
checkAndPrintInfo() {
    getRocmVersion
    getGfxVersion
    getMarketingName
}

# Call the function to check and print all information
checkAndPrintInfo
