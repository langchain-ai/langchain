#!/bin/bash

yum remove openssl-devel -y
yum install openssl11 openssl11-devel -y
# Remove python 3.9 and install python 3.10
yum list
yum remove python39 -y
yum install python310 -y
python3 --version

cd ..
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r vercel_requirements.txt
python3 -c "from requests import HTTPError, Response; print(HTTPError, Response)"
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
python3 generate_api_reference_links.py
