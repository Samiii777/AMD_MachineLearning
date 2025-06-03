#!/bin/bash
set -e

LOG_FILE="log.txt"

# Global configuration variables
ROCM_VERSION=""
UBUNTU_VERSION=""
PYTHON_VERSION=""
PIP_FLAGS=""
DRIVER_URL=""
TORCH_URL=""
TORCHVISION_URL=""
TRITON_URL=""
TORCHAUDIO_URL=""
TENSORFLOW_URL=""
ONNXRUNTIME_REPO_URL=""
USE_VENV=false
VENV_PATH="./venv"

# Function to log messages
log() {
    local level="$1"
    local message="$2"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message" >> "$LOG_FILE"
}

# Function to detect Ubuntu version and set appropriate pip flags
detect_os() {
    log "INFO" "Detecting operating system..."
    
    # Detect Ubuntu version
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ]; then
            UBUNTU_VERSION=$VERSION_ID
            log "INFO" "Detected Ubuntu $UBUNTU_VERSION"
            
            # Set pip flags based on Ubuntu version
            case "$UBUNTU_VERSION" in
                "22.04")
                    PYTHON_VERSION="cp310"
                    PIP_FLAGS=""
                    log "INFO" "Using Python 3.10 packages"
                    ;;
                "24.04")
                    PYTHON_VERSION="cp312"
                    PIP_FLAGS="--break-system-packages --no-warn-script-location"
                    log "INFO" "Using Python 3.12 packages with --break-system-packages flag"
                    # Ensure ~/.local/bin is in PATH for Ubuntu 24.04
                    ensure_local_bin_in_path
                    ;;
                *)
                    log "WARNING" "Ubuntu version $UBUNTU_VERSION not explicitly supported, defaulting to 22.04 settings"
                    UBUNTU_VERSION="22.04"
                    PYTHON_VERSION="cp310"
                    PIP_FLAGS=""
                    ;;
            esac
        else
            log "ERROR" "This script only supports Ubuntu. Detected OS: $ID"
            exit 1
        fi
    else
        log "ERROR" "Cannot detect operating system. /etc/os-release not found."
        exit 1
    fi
}

# Function to ensure ~/.local/bin is in PATH
ensure_local_bin_in_path() {
    local local_bin="$HOME/.local/bin"
    
    # Check if ~/.local/bin is already in PATH
    if [[ ":$PATH:" != *":$local_bin:"* ]]; then
        log "INFO" "Adding ~/.local/bin to PATH for this session"
        export PATH="$PATH:$local_bin"
        
        # Add to shell profile for persistence
        local shell_profile=""
        if [ -n "$BASH_VERSION" ]; then
            shell_profile="$HOME/.bashrc"
        elif [ -n "$ZSH_VERSION" ]; then
            shell_profile="$HOME/.zshrc"
        else
            shell_profile="$HOME/.profile"
        fi
        
        # Check if the PATH export is not already in the profile
        if [ -f "$shell_profile" ] && ! grep -q "export PATH.*\.local/bin" "$shell_profile"; then
            log "INFO" "Adding ~/.local/bin to PATH in $shell_profile"
            echo "" >> "$shell_profile"
            echo "# Added by ROCm install script" >> "$shell_profile"
            echo "export PATH=\"\$PATH:\$HOME/.local/bin\"" >> "$shell_profile"
            log "INFO" "Please run 'source $shell_profile' or restart your terminal after installation"
        fi
    else
        log "INFO" "~/.local/bin is already in PATH"
    fi
}

# Function to setup virtual environment
setup_virtual_env() {
    if [ "$USE_VENV" = true ]; then
        log "INFO" "Setting up virtual environment at $VENV_PATH"
        check_and_install_package "python3-venv"
        
        if [ -d "$VENV_PATH" ]; then
            log "INFO" "Virtual environment already exists at $VENV_PATH"
        else
            log "INFO" "Creating new virtual environment at $VENV_PATH"
            python3 -m venv "$VENV_PATH"
        fi
        
        # Activate virtual environment
        source "$VENV_PATH/bin/activate"
        log "INFO" "Virtual environment activated"
        
        # Upgrade pip in virtual environment (no --break-system-packages needed in venv)
        pip3 install --upgrade pip
        log "INFO" "Pip upgraded in virtual environment"
    fi
}

# Function to initialize version and load ROCm configuration
init_rocm_config() {
    local version=${1:-"6.4.1"}
    
    # Check if version exists in config
    local version_exists=$(yq e ".versions.\"$version\"" rocm_config.yml)
    if [ "$version_exists" = "null" ]; then
        log "ERROR" "Version $version not found in rocm_config.yml"
        echo "Error: Version $version not found in configuration!" >&2
        echo "Available versions:" >&2
        yq e '.versions | keys | .[]' rocm_config.yml | sed 's/^/  - /' >&2
        echo "Try using one of the versions listed above." >&2
        exit 1
    fi
    
    # Check if Ubuntu version is supported for this ROCm version
    local ubuntu_config=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\"" rocm_config.yml)
    if [ "$ubuntu_config" = "null" ]; then
        log "ERROR" "Ubuntu $UBUNTU_VERSION not supported for ROCm version $version"
        echo "Error: Ubuntu $UBUNTU_VERSION not supported for ROCm version $version!" >&2
        echo "Supported Ubuntu versions for ROCm $version:" >&2
        yq e ".versions.\"$version\".ubuntu | keys | .[]" rocm_config.yml | sed 's/^/  - /' >&2
        exit 1
    fi
    
    # Read configuration for the specified version and Ubuntu version
    ROCM_VERSION="$version"
    DRIVER_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".rocm.driver_url" rocm_config.yml)

    # Set PyTorch package wheel URLs as global variables
    TORCH_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".pytorch.wheel_urls.torch" rocm_config.yml)
    TORCHVISION_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".pytorch.wheel_urls.torchvision" rocm_config.yml)
    TRITON_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".pytorch.wheel_urls.triton" rocm_config.yml)
    TORCHAUDIO_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".pytorch.wheel_urls.torchaudio" rocm_config.yml)

    # Set TensorFlow and ONNX Runtime URLs as global variables
    TENSORFLOW_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".tensorflow.wheel_url" rocm_config.yml)
    ONNXRUNTIME_REPO_URL=$(yq e ".versions.\"$version\".ubuntu.\"$UBUNTU_VERSION\".onnxruntime.repo_url" rocm_config.yml)
    
    log "INFO" "Successfully loaded configuration for ROCm version $version on Ubuntu $UBUNTU_VERSION"
}

# Function to check and install a package if not present
check_and_install_package() {
    local package_name="$1"
    if dpkg-query -W -f='${db:Status-Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
        log "INFO" "$package_name is already installed."
    else
        log "INFO" "$package_name is not installed. Installing..."
        sudo apt update
        sudo apt install "$package_name" -y
        log "INFO" "$package_name has been successfully installed."
    fi
}

# Function to check if a pip package is installed and install it if not
check_and_install_pip_package() {
    local package_name="$1"
    
    if [ "$USE_VENV" = true ]; then
        # We're in an activated virtual environment, so use regular pip3 command
        if pip3 list | grep "^$package_name " > /dev/null; then
            log "INFO" "$package_name is already installed."
        else
            log "INFO" "$package_name not found, installing..."
            pip3 install "$package_name"
            log "INFO" "$package_name has been installed."
        fi
    else
        # Use system pip with appropriate flags
        if pip3 list | grep "^$package_name " > /dev/null; then
            log "INFO" "$package_name is already installed."
        else
            log "INFO" "$package_name not found, installing..."
            pip3 install "$package_name" $PIP_FLAGS
            log "INFO" "$package_name has been installed."
        fi
    fi
}

# Function to install yq for parsing YAML
install_yq() {
    if ! command -v yq &> /dev/null; then
        log "INFO" "yq could not be found, installing..."
        sudo snap install yq
    fi
}

# Function to install AMD Driver
install_amd_driver() {
    # Check if amdgpu-uninstall exists
    if ! command -v amdgpu-uninstall &> /dev/null
    then
        log "INFO" "amdgpu-uninstall not found, skipping uninstallation..."
    else
       log "INFO" "Uninstalling existing AMD GPU drivers..."
        amdgpu-uninstall -y
    fi

    log "INFO" "Downloading and installing AMD Driver..."
    wget -P /tmp/ "$DRIVER_URL"
    sudo apt install -y /tmp/$(basename "$DRIVER_URL")
    amdgpu-install -y --usecase=graphics,rocm

    # Add user to render and video group
    sudo usermod -aG render "$(whoami)"
    sudo usermod -aG video "$(whoami)"
}

# Function to install PyTorch and related packages
install_pytorch() {
    WHEEL_DIR="/tmp/wheels"
    mkdir -p "$WHEEL_DIR"

    local packages=("torch" "torchvision" "triton" "torchaudio")
    local urls=("$TRITON_URL" "$TORCH_URL" "$TORCHVISION_URL" "$TORCHAUDIO_URL")

    for url in "${urls[@]}"; do
        if [ -z "$url" ]; then
            log "ERROR" "One or more package URLs are not set. Exiting."
            return 1
        fi
    done

    local wheel_files=()
    for i in "${!packages[@]}"; do
        package="${packages[$i]}"
        wheel_url="${urls[$i]}"
        wheel_file="$WHEEL_DIR/$(basename "$wheel_url")"

        # Download the wheel file if it doesn't already exist
        if [ ! -f "$wheel_file" ]; then
            log "INFO" "Downloading $package wheel from $wheel_url..."
            wget -O "$wheel_file" "$wheel_url" || {
                log "ERROR" "Failed to download $package from $wheel_url."
                return 1
            }
        else
            log "INFO" "$package wheel already exists at $wheel_file. Skipping download."
        fi

        # Add the wheel file to the list for batch installation
        wheel_files+=("$wheel_file")
    done

    log "INFO" "Installing all packages from downloaded wheels..."
    if [ "$USE_VENV" = true ]; then
        pip3 install "${wheel_files[@]}" --force-reinstall || {
            log "ERROR" "Failed to install one or more packages."
            return 1
        }
    else
        pip3 install "${wheel_files[@]}" --force-reinstall $PIP_FLAGS || {
            log "ERROR" "Failed to install one or more packages."
            return 1
        }
    fi

    log "INFO" "All packages installed successfully."
}

# Function to install TensorFlow
install_tensorflow() {
    log "INFO" "Installing TensorFlow from $TENSORFLOW_URL..."
    if [ "$USE_VENV" = true ]; then
        pip3 install "$TENSORFLOW_URL"
        pip3 install tf-keras --no-deps
    else
        pip3 install "$TENSORFLOW_URL" $PIP_FLAGS
        pip3 install tf-keras --no-deps $PIP_FLAGS
    fi
    log "INFO" "TensorFlow installed."
}

# Function to install ONNX Runtime
install_onnx_runtime() {
    log "INFO" "Installing ONNX Runtime from $ONNXRUNTIME_REPO_URL..."
    check_and_install_package "migraphx"
    check_and_install_package "half"
    
    if pip3 list | grep -E "onnxruntime(-rocm)?|onnxruntime?|onnxruntime(-gpu)$|^onnx$"; then
        log "INFO" "Found existing ONNX Runtime installation, uninstalling..."
        if [ "$USE_VENV" = true ]; then
            pip3 uninstall onnxruntime-rocm onnxruntime -y
        else
            pip3 uninstall onnxruntime-rocm onnxruntime -y $PIP_FLAGS
        fi
    fi
    
    if [ "$USE_VENV" = true ]; then
        pip3 install onnxruntime-rocm -f "$ONNXRUNTIME_REPO_URL"
    else
        pip3 install onnxruntime-rocm -f "$ONNXRUNTIME_REPO_URL" $PIP_FLAGS
    fi
    
    if pip3 list | grep -E "onnxruntime(-rocm)?"; then
        log "INFO" "ONNX Runtime is successfully installed."
    fi
}

# Function to check ROCm driver status
check_rocm_status() {
    log "INFO" "Checking ROCm driver status..."
    
    # Check if ROCm devices are available
    if [ -d "/sys/class/drm" ]; then
        local amd_devices=$(find /sys/class/drm -name "card*" -exec grep -l "^DRIVER=amdgpu" {}/device/uevent \; 2>/dev/null | wc -l)
        if [ "$amd_devices" -gt 0 ]; then
            echo "✅ AMD GPU devices detected: $amd_devices"
            log "INFO" "AMD GPU devices detected: $amd_devices"
            
            # Show GPU details
            echo "📊 GPU Details:"
            for card in /sys/class/drm/card*; do
                if [ -f "$card/device/uevent" ] && grep -q "^DRIVER=amdgpu" "$card/device/uevent"; then
                    local pci_id=$(grep "^PCI_ID=" "$card/device/uevent" | cut -d'=' -f2)
                    echo "  - $(basename $card): PCI ID $pci_id"
                fi
            done
        else
            echo "❌ No AMD GPU devices found"
            log "WARNING" "No AMD GPU devices found"
        fi
    fi
    
    # Check render devices (these are needed for compute)
    local render_devices=$(ls /sys/class/drm/renderD* 2>/dev/null | wc -l)
    if [ "$render_devices" -gt 0 ]; then
        echo "✅ Render devices available: $render_devices"
        log "INFO" "Render devices available: $render_devices"
    else
        echo "❌ No render devices found"
        log "WARNING" "No render devices found"
    fi
    
    # Check if user is in render and video groups
    local groups=$(groups)
    if [[ $groups == *"render"* && $groups == *"video"* ]]; then
        echo "✅ User is in render and video groups"
        log "INFO" "User is in render and video groups"
    else
        echo "❌ User not in required groups (render/video)"
        log "WARNING" "User not in required groups. Please run: sudo usermod -aG render,video \$(whoami) && newgrp render"
    fi
    
    # Check if ROCm runtime is available
    if command -v rocm-smi &> /dev/null; then
        echo "✅ ROCm SMI available"
        log "INFO" "ROCm SMI available"
        echo "🔍 ROCm SMI Output:"
        rocm-smi --showproductname 2>/dev/null || echo "❌ ROCm SMI failed - reboot may be required"
    else
        echo "❌ ROCm SMI not found"
        log "WARNING" "ROCm SMI not found"
    fi
    
    echo "📝 If ROCm tests fail, try rebooting the system first"
    echo ""
}

# Function to run tests
run_tests() {
    log "INFO" "Running comprehensive system tests..."
    echo "🧪 Running ROCm ML Framework Tests..."
    echo "=================================="
    
    # Check ROCm status first
    check_rocm_status
    
    # Test PyTorch
    echo "🔥 Testing PyTorch..."
    if python3 tests/test_pytorch.py; then
        echo "✅ PyTorch test completed"
        
        # Show PyTorch version info
        python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('⚠️  GPU not available - check ROCm installation')
"
    else
        echo "❌ PyTorch test failed"
    fi
    echo ""
    
    # Test ONNX Runtime  
    echo "🤖 Testing ONNX Runtime..."
    if python3 tests/test_onnxruntime.py; then
        echo "✅ ONNX Runtime test completed"
    else
        echo "❌ ONNX Runtime test failed"
    fi
    echo ""
    
    # Test TensorFlow
    echo "🧠 Testing TensorFlow..."
    if python3 tests/test_tensorflow.py; then
        echo "✅ TensorFlow test completed"
    else
        echo "❌ TensorFlow test failed"
    fi
    echo ""
    
    echo "🎯 Test Summary:"
    echo "==============="
    echo "Check the log.txt file for detailed test results"
    echo "If GPU tests fail, ensure you have:"
    echo "  1. Rebooted after driver installation"
    echo "  2. Added user to render/video groups"
    echo "  3. AMD GPU compatible with ROCm"
    
    log "INFO" "Tests completed."
}

# Function to prompt the user to reboot or not
prompt_reboot() {
    read -p "Do you want to reboot the system now? (y/n): " choice
    case "$choice" in
      y|Y )
        shutdown -r now ;;
      n|N )
        log "INFO" "Reboot skipped. You may need to restart the system later for changes to take effect." ;;
      * )
        log "INFO" "Invalid choice. Reboot skipped. You may need to restart the system later for changes to take effect." ;;
    esac
}

# Function to check and install all necessary packages
check_and_install_dependencies() {
    install_yq
    check_and_install_package "python3-pip"
    check_and_install_pip_package "wheel"
    check_and_install_pip_package "setuptools"
    check_and_install_package "libstdc++-12-dev"
    check_and_install_package "libclblast-dev"
}

# Function to install requirements.txt
install_requirements() {
    log "INFO" "Installing requirements from requirements.txt..."
    if [ "$USE_VENV" = true ]; then
        pip3 install -r requirements.txt
    else
        pip3 install -r requirements.txt $PIP_FLAGS
    fi
    log "INFO" "Requirements installed."
}

# Function to display installation options
display_menu() {
    echo "Please select an installation option:"
    echo "1. Install everything (driver + all frameworks)"
    echo "2. Install driver only"
    echo "3. Install all ML frameworks (PyTorch, TensorFlow, ONNX Runtime)"
    echo "4. Install PyTorch only"
    echo "5. Install TensorFlow only" 
    echo "6. Install ONNX Runtime only"
    echo "7. Quick system check"
    read -p "Enter your choice (1-7): " MENU_CHOICE
    echo
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --venv)
                USE_VENV=true
                shift
                ;;
            --venv-path)
                if [[ -n "$2" && "$2" != --* ]]; then
                    VENV_PATH="$2"
                    USE_VENV=true
                    shift 2
                else
                    log "ERROR" "Argument for $1 is missing"
                    exit 1
                fi
                ;;
            *)
                # Assume it's a version number
                VERSION_ARG="$1"
                shift
                ;;
        esac
    done
}

# Main function
main() {
    # Parse command line arguments first
    parse_arguments "$@"
    
    # Detect OS and set appropriate flags
    detect_os
    
    check_and_install_dependencies
    
    # Check if a version was provided
    if [ -n "$VERSION_ARG" ]; then
        log "INFO" "Checking for ROCm version: $VERSION_ARG"
        # This will exit if version is not found
        init_rocm_config "$VERSION_ARG"
    else
        # Use default version from metadata if available
        local default_version=$(yq e ".metadata.default_version" rocm_config.yml)
        if [ -n "$default_version" ] && [ "$default_version" != "null" ]; then
            log "INFO" "No version specified, using default: $default_version"
            init_rocm_config "$default_version"
        else
            log "INFO" "Using hardcoded default version: 6.4.1"
            init_rocm_config "6.4.1"
        fi
    fi
    
    # Setup virtual environment if requested
    if [ "$USE_VENV" = true ]; then
        setup_virtual_env
        log "INFO" "Using virtual environment at $VENV_PATH"
    fi
    
    # If we get here, version check passed
    log "INFO" "Version check passed, displaying menu"
    
    display_menu
    
    case $MENU_CHOICE in
        1)
            install_amd_driver
            install_pytorch
            install_tensorflow
            install_onnx_runtime
            run_tests
            install_requirements
            prompt_reboot
            ;;
        2)
            install_amd_driver
            ;;
        3)
            install_pytorch
            install_tensorflow
            install_onnx_runtime
            run_tests
            install_requirements
            ;;
        4)
            install_pytorch
            python3 tests/test_pytorch.py
            install_requirements
            ;;
        5)
            install_tensorflow
            python3 tests/test_tensorflow.py
            install_requirements
            ;;
        6)
            install_onnx_runtime
            python3 tests/test_onnxruntime.py
            install_requirements
            ;;
        7)
            run_tests
            ;;
        *)
            log "ERROR" "Invalid option selected. Exiting..."
            exit 1
            ;;
    esac
    
    # Deactivate virtual environment if it was used
    if [ "$USE_VENV" = true ]; then
        deactivate 2>/dev/null || true
        log "INFO" "Virtual environment deactivated"
    fi
}

main "$@"
