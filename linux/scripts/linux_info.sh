#!/bin/bash

# Function to extract Linux distribution information
getLinuxDistribution() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ]; then
            echo "Linux Distribution: Ubuntu $VERSION"
            echo "Code Name: $VERSION_CODENAME"
        else
            echo "Linux Distribution: $PRETTY_NAME"
        fi
    else
        echo "Unable to determine Linux Distribution."
    fi
}

# Function to extract kernel version
getKernelVersion() {
    local kernelVersion=$(uname -r)
    if [ -n "$kernelVersion" ]; then
        echo "Kernel Version: $kernelVersion"
    else
        echo "Unable to determine Kernel Version."
    fi
}

# Function to extract system uptime
getSystemUptime() {
    local uptime=$(uptime -p)
    if [ -n "$uptime" ]; then
        echo "System Uptime: $uptime"
    else
        echo "Unable to determine System Uptime."
    fi
}

# Function to extract system architecture
getSystemArchitecture() {
    local architecture=$(uname -m)
    if [ -n "$architecture" ]; then
        echo "System Architecture: $architecture"
    else
        echo "Unable to determine System Architecture."
    fi
}

# Function to extract hostname
getHostname() {
    local hostname=$(hostname)
    if [ -n "$hostname" ]; then
        echo "Hostname: $hostname"
    else
        echo "Unable to determine Hostname."
    fi
}

# Function to check and print all information
checkAndPrintInfo() {
    getLinuxDistribution
    getKernelVersion
    getSystemUptime
    getSystemArchitecture
    getHostname
}

# Call the function to check and print all information
checkAndPrintInfo
