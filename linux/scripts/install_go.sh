#!/bin/bash

# Default Go version
DEFAULT_GO_VERSION="1.22.0"

# Use user-provided Go version if provided, otherwise use default
GO_VERSION=${1:-$DEFAULT_GO_VERSION}

# Download and install Go

GO_URL="https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
wget "${GO_URL}"
if [ $? -ne 0 ]; then
    echo "Error downloading Go. Exiting."
    exit 1
fi

if [ -d /usr/local/go ]; then
    sudo rm -rf /usr/local/go
fi

sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz

# Add Go binary path to the PATH environment variable
export PATH=$PATH:/usr/local/go/bin
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# Clean up downloaded file
rm -f go${GO_VERSION}.linux-amd64.tar.gz

# Display Go version
go version


