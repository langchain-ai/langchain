#!/bin/bash

### See: https://github.com/urllib3/urllib3/issues/2168
# Requests lib breaks for old SSL versions,
# which are defaults on Amazon Linux 2 (which Vercel uses for builds)
yum -y update
yum remove openssl-devel -y
yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip -y
amazon-linux-extras install python3.8 -y
yum install openssl11 -y
yum install openssl11-devel -y

# install quarto
wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz
tar -xvzf quarto-1.3.450-linux-amd64.tar.gz
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
cp ../.github/CONTRIBUTING.md docs/contributing.md
wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O docs/langserve.md
quarto render docs/
python3.8 scripts/generate_api_reference_links.py
