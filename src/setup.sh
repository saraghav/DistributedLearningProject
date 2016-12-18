#!/bin/bash

apt-get --assume-yes install python3 python3-pip python3-numpy python3-scipy python3-matplotlib
apt-get --assume-yes install screen

# Ubuntu/Linux 64-bit, CPU only, Python 3.4
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp34-cp34m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
