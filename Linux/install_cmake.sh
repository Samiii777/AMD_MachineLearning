#Replace the version with your desired one.

sudo apt autoremove cmake
sudo apt clean
wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.tar.gz
tar xvf cmake-3.26.3.tar.gz
cd cmake-3.26.3
./bootstrap && make && sudo make install