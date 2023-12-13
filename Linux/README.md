# Linux
Everything you need to setup on your Linux system for Machine Learning Stuff with AMD GPU

## Install AMD Driver with ROCm

1. Run below commands manually in terminal

```
sudo usermod -aG render $LOGNAME
sudo usermod -aG video $LOGNAME
```
2. Then Simply run below bash file
```
sudo sh ./install_amd_driver_on_ubuntu.sh
```