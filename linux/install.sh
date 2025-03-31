#!/bin/bash
set -e

LOG_FILE="log.txt"

# Global configuration variables
ROCM_VERSION=""
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
        
        # Upgrade pip in virtual environment
        pip3 install --upgrade pip
        log "INFO" "Pip upgraded in virtual environment"
    fi
}

# Function to initialize version and load ROCm configuration
init_rocm_config() {
    local version=${1:-"6.3.4"}
    
    # Check if version exists in config - yq returns "null" for non-existent keys, not an error code
    local version_exists=$(yq e ".versions.\"$version\"" rocm_config.yml)
    if [ "$version_exists" = "null" ]; then
        log "ERROR" "Version $version not found in rocm_config.yml"
        echo "Error" "Version $version not found in configuration!" >&2
        echo "Available versions:" >&2
        yq e '.versions | keys | .[]' rocm_config.yml | sed 's/^/  - /' >&2
        echo "Try using one of the versions listed above." >&2
        exit 1  # Exit immediately
    fi
    
    # Read configuration for the specified version
    ROCM_VERSION=$(yq e ".versions.\"$version\".rocm.version" rocm_config.yml)
    DRIVER_URL=$(yq e ".versions.\"$version\".rocm.driver_url" rocm_config.yml)

    # Set PyTorch package wheel URLs as global variables
    TORCH_URL=$(yq e ".versions.\"$version\".pytorch.wheel_urls.torch" rocm_config.yml)
    TORCHVISION_URL=$(yq e ".versions.\"$version\".pytorch.wheel_urls.torchvision" rocm_config.yml)
    TRITON_URL=$(yq e ".versions.\"$version\".pytorch.wheel_urls.triton" rocm_config.yml)
    TORCHAUDIO_URL=$(yq e ".versions.\"$version\".pytorch.wheel_urls.torchaudio" rocm_config.yml)

    # Set TensorFlow and ONNX Runtime URLs as global variables
    TENSORFLOW_URL=$(yq e ".versions.\"$version\".tensorflow.wheel_url" rocm_config.yml)
    ONNXRUNTIME_REPO_URL=$(yq e ".versions.\"$version\".onnxruntime.repo_url" rocm_config.yml)
    
    log "INFO" "Successfully loaded configuration for ROCm version $version"
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
        # Use system pip
        if pip3 list | grep "^$package_name " > /dev/null; then
            log "INFO" "$package_name is already installed."
        else
            log "INFO" "$package_name not found, installing..."
            pip3 install "$package_name"
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
    pip3 install "${wheel_files[@]}" --force-reinstall || {
        log "ERROR" "Failed to install one or more packages."
        return 1
    }

    log "INFO" "All packages installed successfully."
}

# Function to install TensorFlow
install_tensorflow() {
    log "INFO" "Installing TensorFlow from $TENSORFLOW_URL..."
    pip3 install "$TENSORFLOW_URL"
    pip3 install tf-keras --no-deps
    log "INFO" "TensorFlow installed."
}

# Function to install ONNX Runtime
install_onnx_runtime() {
    log "INFO" "Installing ONNX Runtime from $ONNXRUNTIME_REPO_URL..."
    check_and_install_package "migraphx"
    check_and_install_package "half"
    
    if pip3 list | grep -E "onnxruntime(-rocm)?|onnxruntime?|onnxruntime(-gpu)$|^onnx$"; then
        log "INFO" "Found existing ONNX Runtime installation, uninstalling..."
        pip3 uninstall onnxruntime-rocm onnxruntime -y
    fi
    pip3 install onnxruntime-rocm -f "$ONNXRUNTIME_REPO_URL"
    if pip3 list | grep -E "onnxruntime(-rocm)?"; then
        log "INFO" "ONNX Runtime is successfully installed."
    fi
}

# Function to run tests
run_tests() {
    log "INFO" "Running tests..."
    python3 tests/test_pytorch.py
    python3 tests/test_onnxruntime.py 
    python3 tests/test_tensorflow.py
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
    pip3 install -r requirements.txt
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
            log "INFO" "Using hardcoded default version: 6.3.4"
            init_rocm_config "6.3.4"
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
