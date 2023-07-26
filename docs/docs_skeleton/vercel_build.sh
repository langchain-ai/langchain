#!/bin/bash

### See: https://github.com/urllib3/urllib3/issues/2168
# Requests lib breaks for old SSL versions,
# which are defaults on Amazon Linux 2 (which Vercel uses for builds)
yum -y update
yum remove openssl-devel -y
yum install gcc bzip2-devel libffi-devel zlib-devel wget tar -y
yum install openssl11 -y
yum install openssl11-devel -y
# Install python 3.11 to connect with openSSL 1.1.1
wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz 
tar xzf Python-3.11.4.tgz 
cd Python-3.11.4 
./configure 
make altinstall
# Check python version
echo "Python Version"
python3.11 --version
cd ..
###

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
