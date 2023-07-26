#!/bin/bash
yum -y update
yum remove openssl-devel -y
yum install gcc bzip2-devel libffi-devel zlib-devel wget tar -y
# Make sure openssl11 is installed before Python compilation
yum install openssl11 -y
yum install openssl11-devel -y

# Locate openssl 1.1.1 library and headers
# OPENSSL_LIB_PATH=$(dirname $(find / -name 'libssl.so.*' | grep 'openssl11'))
# OPENSSL_INCLUDE_PATH=$(dirname $(find / -name 'openssl' | grep 'openssl11'))

echo "OPENSSL VERSION"
openssl11 version 

# Install python 3.11 to connect with openSSL 1.1.1
wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz 
tar xzf Python-3.11.4.tgz 
cd Python-3.11.4 
./configure 
#--with-openssl=${OPENSSL_LIB_PATH} CPPFLAGS="-I${OPENSSL_INCLUDE_PATH}"
make altinstall

# Check python version
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
