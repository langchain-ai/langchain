#!/bin/bash

echo "In custom script!"
cd ..
python3 --version
pip3 --version
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
mkdir -p docs_skeleton/static/api_reference
cd api_reference
make html
cp -r _build/* ../docs_skeleton/static/api_reference
cd ..
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
