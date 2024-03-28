#!/bin/sh

# Function to convert memory size to GB
convert_to_gb() {
    awk '{printf "%.2f\n", $1/1024/1024/1024}'
}

# Counter for GPU number
gpu_number=0
total_vram_used=0

# Get GPU addresses
gpu_addresses=$(lspci | grep VGA | awk '{print $1}')

# Iterate over each GPU address
for gpu_address in $gpu_addresses; do
    echo "GPU $gpu_number Address: $gpu_address"
    
    # Fetch detailed information for the GPU using lspci -vv -s <gpu_address>
    gpu_info=$(sudo lspci -vv -s "$gpu_address")
    
    # Extract relevant information about PCIe link width
    link_width=$(echo "$gpu_info" | grep -oP 'LnkSta:.*Width \K[^ ]+')


    # Print PCIe link width
    echo "PCIe Link Width: $link_width"

    # Convert GPU address to the required format for accessing memory information
    converted_gpu_address="0000:${gpu_address}"
    gpu_info_folder="/sys/bus/pci/devices/$converted_gpu_address"

    # Form the file path for memory information
    memory_info_file_total="$gpu_info_folder/mem_info_vram_total"

    # Check if the memory info file exists
    if [ -f "$memory_info_file_total" ]; then
        # Read memory size from the file and convert to GB
        memory_size=$(sudo cat "$memory_info_file_total" | convert_to_gb)
        echo "Memory Size: $memory_size GB"
    else
        echo "Memory Info File not found!"
    fi

    # Form the file path for used memory information
    memory_info_file_used="$gpu_info_folder/mem_info_vram_used"

    # Check if the used memory info file exists
    if [ -f "$memory_info_file_used" ]; then
        # Read used memory size from the file and convert to GB
        vram_used=$(sudo cat "$memory_info_file_used" | convert_to_gb)
        echo "VRAM Used: $vram_used GB/$memory_size GB"

        # Accumulate the VRAM used across all GPUs
        total_vram_used=$(awk "BEGIN {printf \"%.2f\", $total_vram_used + $vram_used}")
    else
        echo "Used Memory Info File not found!"
    fi

    echo "---------------------------------------------"
    
    # Increment GPU number
    gpu_number=$(expr $gpu_number + 1)
done

# Print total VRAM used across all GPUs
echo "Total VRAM Used Across All GPUs: $total_vram_used GB"
