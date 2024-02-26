#!/bin/bash

# Download and install Go
GO_VERSION="1.22.0"
GO_URL="https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"

# Download and install Go
wget "${GO_URL}"
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz

# Add Go binary path to the PATH environment variable
export PATH=$PATH:/usr/local/go/bin

# Clean up downloaded file
rm -f go${GO_VERSION}.linux-amd64.tar.gz

# Display Go version
go version
