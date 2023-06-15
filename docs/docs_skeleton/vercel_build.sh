#!/bin/bash

cd ..

curl https://pyenv.run | bash
eval "$(pyenv init -)"
exec $SHELL
pyenv install 3.10
pyenv global 3.10
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p docs_skeleton/static/api_reference
cd api_reference
make html
cp -r _build/* ../docs_skeleton/static/api_reference
cd ..
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
