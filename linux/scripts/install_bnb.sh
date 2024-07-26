#!/bin/bash

# Define variables
REPO_URL="https://github.com/ROCm/bitsandbytes"
BRANCH_NAME="rocm_enabled"
REPO_DIR="bitsandbytes"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to activate virtual environment
activate_venv() {
    local venv_dir="$1"
    if [ -f "$venv_dir/bin/activate" ]; then
        echo "Activating virtual environment in $venv_dir..."
        . "$venv_dir/bin/activate"
    else
        echo "Virtual environment not found in $venv_dir."
        return 1
    fi
}

# Function to create a virtual environment
create_venv() {
    local venv_dir="$1"
    echo "Creating virtual environment in $venv_dir..."
    python3 -m venv "$venv_dir" || { echo "Failed to create virtual environment." >&2; exit 1; }
    activate_venv "$venv_dir" || { echo "Failed to activate virtual environment." >&2; exit 1; }
}

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --venv)
            VENV_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "Usage: $0 [--venv <venv_directory>]" >&2
            exit 1
            ;;
    esac
done

# Handle virtual environment
if [ -n "$VENV_DIR" ]; then
    if [ -d "$VENV_DIR" ]; then
        activate_venv "$VENV_DIR" || create_venv "$VENV_DIR"
    else
        create_venv "$VENV_DIR"
    fi
fi

# Clone the repository
if [ -d "$REPO_DIR" ]; then
    echo "Directory $REPO_DIR already exists. Pulling the latest changes..."
    cd "$REPO_DIR" || exit 1
    git pull origin "$BRANCH_NAME" || exit 1
else
    git clone --recurse "$REPO_URL" "$REPO_DIR" || exit 1
    cd "$REPO_DIR" || exit 1
fi

# Checkout the specified branch
git checkout "$BRANCH_NAME" || exit 1

# Install Python dependencies
if [ -n "$VENV_DIR" ]; then
    pip install -r requirements-dev.txt || exit 1
else
    python3 -m pip install -r requirements-dev.txt || exit 1
fi

# Configure and build with cmake
cmake -DCOMPUTE_BACKEND=hip -S . || exit 1

# Build the project
make || exit 1

# Install the package
if [ -n "$VENV_DIR" ]; then
    pip install . || exit 1
else
    python3 -m pip install . || exit 1
fi

echo "Script completed successfully."
