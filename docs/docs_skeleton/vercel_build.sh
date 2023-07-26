#!/bin/bash

yum remove openssl-devel -y
yum install openssl11 openssl11-devel -y
yum update -y
yum install python3.11 -y
python3.11 --version

cd ..
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r vercel_requirements.txt
python3.11 -c "from requests import HTTPError, Response; print(HTTPError, Response)"
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
python3.11 generate_api_reference_links.py
