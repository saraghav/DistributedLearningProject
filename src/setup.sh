#!/bin/bash
if [ -d "/mydata" ]; then
    chmod 777 /mydata
    chown -R saraghav /mydata
    chgrp -R wingslab-PG0 /mydata
fi


apt-get --assume-yes update
apt-get --assume-yes install postfix
apt-get --assume-yes install screen
apt-get --assume-yes install vim
apt-get --assume-yes install python3 python3-pip python3-numpy python3-scipy python3-matplotlib

# Ubuntu/Linux 64-bit, CPU only, Python 3.4
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp34-cp34m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
