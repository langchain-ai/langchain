#!/bin/bash

uname -or
uname -a

cd ..
yum install gcc make patch zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl11-devel tk-devel libffi-devel xz-devel
	
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.9
pyenv global 3.9
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdir -p docs_skeleton/static/api_reference
cd api_reference
make html
cp -r _build/* ../docs_skeleton/static/api_reference
cd ..
cp -r extras/* docs_skeleton/docs
cd docs_skeleton
nbdoc_build
