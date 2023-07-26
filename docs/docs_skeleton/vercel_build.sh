#!/bin/bash

yum install gcc bzip2-devel libffi-devel zlib-devel wget tar -y
# Make sure openssl-devel is installed before Python compilation
yum install openssl-devel -y
yum install wget tar -y

# Install python 3.11 to connect with openSSL 1.1.1
wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz 
tar xzf Python-3.11.4.tgz 
cd Python-3.11.4 
./configure
make altinstall
echo "Python Version"
python3.11 --version
cd ..

# Install nbdev and generate docs
cd ..
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r vercel_requirements.txt
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
python3.11 generate_api_reference_links.py
