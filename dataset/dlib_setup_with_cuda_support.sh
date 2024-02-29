#!/bin/bash

# Uninstall existing dlib if already installed
pip uninstall dlib -y

# Clone dlib repository
git clone https://github.com/davisking/dlib.git

# Navigate to dlib directory
cd dlib

# Create build directory
mkdir build

# Navigate to build directory
cd build

# Configure dlib build with CUDA support and AVX instructions
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1

# Build dlib
cmake --build .

# Navigate back to dlib directory
cd ..

# Install dlib
python setup.py install
