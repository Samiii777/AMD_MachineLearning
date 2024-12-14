#!/usr/bin/env bash


# Function to check if tinygrad is installed
check_tinygrad() {
    if pip show tinygrad &> /dev/null; then
        version=$(pip show tinygrad | grep Version | cut -d ' ' -f 2)
        echo "tinygrad version $version is already installed."
        return 0
    else
        return 1
    fi
}

# Check for -y parameter
force_install=false
if [[ "$1" == "-y" ]]; then
    force_install=true
    shift
fi

# Check if tinygrad is already installed
if check_tinygrad; then
    if ! $force_install; then
        echo "Exiting. Use -y parameter to force reinstallation."
        exit 0
    fi
fi

# Set default installation directory
DEFAULT_DIR="$(dirname "$0")/tinygrad"

# Use the provided argument or default to the script's directory
INSTALL_DIR="${1:-$DEFAULT_DIR}"

# Install git if not already installed
command -v git >/dev/null 2>&1 || { echo >&2 "Git is required but not installed. Aborting."; exit 1; }

# Create the installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Clone the tinygrad repository
git clone https://github.com/tinygrad/tinygrad.git "$INSTALL_DIR"

# Change to the tinygrad directory
cd "$INSTALL_DIR"

# Install tinygrad
python3 -m pip install -e .

# Check if tinygrad is installed
if check_tinygrad; then
    echo "tinygrad has been successfully installed in $INSTALL_DIR"
else
    echo "Failed to install tinygrad. Please check for errors."
fi
