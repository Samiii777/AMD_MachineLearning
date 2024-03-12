#!/bin/bash

# Function to extract CPU model
getCpuModel() {
    local cpuModel=$(lscpu | sed -n '/Model name/s/^[^:]*:[[:space:]]*//p')

    # Check and print GPU version
    if [ -n "$cpuModel" ]; then
        echo "CPU Model: $cpuModel"
    else
        echo "Unable to determine CPU model."
    fi
}

# Function to extract Marketing Name
getCpuArch() {
    local cpuArch=$(lscpu | sed -n '/Architecture/s/^[^:]*:[[:space:]]*//p')
    # Check and print Marketing Name
    if [ -n "$cpuArch" ]; then
        echo "CPU Architecture: $cpuArch"
    else
        echo "Unable to determine CPU Architecture."
    fi
}

# Function to check and print all information
checkAndPrintInfo() {
    getCpuModel
    getCpuArch
}

# Call the function to check and print all information
checkAndPrintInfo
