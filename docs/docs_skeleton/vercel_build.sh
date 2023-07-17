#!/bin/bash

cd ..
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp -r extras/* docs_skeleton/docs
python3 docs_skeleton/link_generator.py
cd docs_skeleton
nbdoc_build
