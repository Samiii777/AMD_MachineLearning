#!/bin/bash

# Check if the script is running with root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script with root privileges (sudo)."
    exit 1
fi

# Update and Install Linux to the Latest Version
apt-get update
apt-get dist-upgrade -y

# Install SSH Server
apt install -y openssh-server

# Install net-tools
apt install -y net-tools

# Install pip3
apt install -y python3-pip 

#Install Jupyter-lab
snap install jupyter
apt install -y jupyter-core

export PATH="$HOME/.local/bin:$PATH"

# Install Git
apt install -y git
apt install -y git-lfs

# Install Libstdc++-12-dev that is required for pytorch
apt install -y libstdc++-12-dev

# Install LightDM without prompts
export DEBIAN_FRONTEND=noninteractive

apt-get install -y lightdm

unset DEBIAN_FRONTEND

# Prompt the user to reboot or not
read -p "Do you want to reboot the system now? (y/n): " choice
case "$choice" in
  y|Y )
    shutdown -r now ;;
  n|N )
    echo "Reboot skipped. You may need to restart the system later for changes to take effect." ;;
  * )
    echo "Invalid choice. Reboot skipped. You may need to restart the system later for changes to take effect." ;;
esac
