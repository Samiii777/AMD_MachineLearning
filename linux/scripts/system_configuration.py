import subprocess
import re

def detect_gpus():
    try:
        # Get PCI devices info using lspci
        output = subprocess.check_output(['lspci', '-nnk']).decode().lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"error": "lspci command not available (likely not a Linux system)"}

    # Vendor ID to name mapping (common GPUs)
    vendor_map = {
        '10de': 'NVIDIA',
        '1002': 'AMD',
        '8086': 'Intel'
    }

    # Extract GPU PCI entries (VGA/3D controllers)
    gpu_lines = [
        line for line in output.split('\n')
        if 'vga' in line or '3d' in line
    ]

    # Parse vendor IDs from PCI IDs (e.g., [10de:2206])
    vendor_ids = []
    for line in gpu_lines:
        match = re.search(r'\[([0-9a-f]{4}):', line)
        if match:
            vendor_ids.append(match.group(1))

    # Map IDs to names and count
    gpu_vendors = [vendor_map.get(vid, 'Unknown') for vid in vendor_ids]
    nvidia_count = gpu_vendors.count('NVIDIA')
    amd_count = gpu_vendors.count('AMD')
    total_gpus = len(gpu_vendors)

    return {
        "total_gpus": total_gpus,
        "nvidia_gpus": nvidia_count,
        "amd_gpus": amd_count,
        "configuration": "Single-GPU" if total_gpus == 1 else "Multi-GPU",
        "vendor_list": gpu_vendors
    }

if __name__ == "__main__":
    gpu_info = detect_gpus()
    if "error" in gpu_info:
        print(gpu_info["error"])
    else:
        print(f"GPUs Detected: {gpu_info['total_gpus']}")
        print(f"Configuration: {gpu_info['configuration']}")
        print(f"NVIDIA GPUs: {gpu_info['nvidia_gpus']}")
        print(f"AMD GPUs: {gpu_info['amd_gpus']}")
        print(f"All Vendors: {', '.join(gpu_info['vendor_list'])}")
