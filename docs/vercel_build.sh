#!/bin/bash

yum -y update
yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip -y
amazon-linux-extras install python3.8 -y

# install quarto
wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz
tar -xzf quarto-1.3.450-linux-amd64.tar.gz
export PATH=$PATH:$(pwd)/quarto-1.3.450/bin/


python3.8 -m venv .venv
source .venv/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install -r vercel_requirements.txt
python3.8 scripts/model_feat_table.py
mkdir docs/templates
cp ../templates/docs/INDEX.md docs/templates/index.md
python3.8 scripts/copy_templates.py
cp ../cookbook/README.md src/pages/cookbook.mdx
wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O docs/langserve.md
wget -q https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md -O docs/langgraph.md
quarto render docs/
