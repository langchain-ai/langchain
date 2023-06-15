#!/bin/bash

uname -or
uname -a

cd ..
sudo yum install gcc openssl-devel bzip2-devel libffi-devel 
wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz 
sudo tar xzf Python-3.11.3.tgz 
cd Python-3.11.3 
sudo ./configure --enable-optimizations 
sudo make altinstall 
sudo rm -f /opt/Python-3.11.3.tgz 

python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install -r requirements.txt
mkdir -p docs_skeleton/static/api_reference
cd api_reference
make html
cp -r _build/* ../docs_skeleton/static/api_reference
cd ..
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
