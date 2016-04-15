# OpenCV 3.0.0 and Python 3 environment setup script
# Based on: http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/
cd ~
rm -rf *

# Ensure system is up-to-date
sudo apt-get update
sudo apt-get upgrade

# Install required packages
sudo apt-get install vim
sudo apt-get install build-essential cmake git pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

# Create server environment
mkdir server

# Install OpenCV 3.0.0
cd~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.0.0
cd ~
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.0.0
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..
make
sudo make install
sudo ldconfig

# Install Boost.Python
sudo apt-get install libboost-all-dev

# Install node
sudo apt-get install nodejs-legacy
# Install Naked framework
pip install Naked

