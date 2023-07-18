#!/bin/bash

cd ..
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
