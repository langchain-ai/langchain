#!/bin/bash

yum remove openssl-devel -y
yum install openssl11 openssl11-devel -y

cd ..
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r vercel_requirements.txt
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
python3 generate_api_reference_links.py
