# Linux
Everything you need to setup on your Linux system for Machine Learning Stuff with AMD GPU

## Install AMD Driver with ROCm

1. Run below commands manually in terminal and reboot the system

```
sudo usermod -aG render $LOGNAME
sudo usermod -aG video $LOGNAME
```
2. Then Simply run below bash file from the corresponding desired ROCm version folder.
```
sudo sh ./install_amd_driver_with_rocm_on_ubuntu.sh
```
