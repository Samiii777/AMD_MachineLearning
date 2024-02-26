# AMD_MachineLearning
Everything you need to setup on your AMD system for Machine Learning Stuff

# [Linux](https://github.com/Samiii777/AMD_MachineLearning/tree/main/linux)

## Install AMD Driver with ROCm

1. First we need to update our Kernel Version to the latest one
```
sudo apt update
sudo apt dist-upgrade -y
```
2. Run below commands manually in terminal and reboot the system
```
sudo usermod -aG render $LOGNAME
sudo usermod -aG video $LOGNAME
```
3. Then Simply run below bash file from the corresponding desired ROCm version folder
```
cd ROCm-6.0.2
sudo sh ./install_amd_driver_with_rocm_on_ubuntu.sh
```
4. To install Pytorch
```
sh ./install_pyTorch.sh
```
5. To install OnnxRuntime (will install MIGraphX as well)
```
sh ./install_OnnxRT.sh
```
