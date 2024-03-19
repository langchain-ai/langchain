#!/bin/bash

yum -y update
yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip -y

# install quarto
wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz
tar -xzf quarto-1.3.450-linux-amd64.tar.gz
export PATH=$PATH:$(pwd)/quarto-1.3.450/bin/


# setup python env
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r vercel_requirements.txt

# autogenerate integrations tables
python3 scripts/model_feat_table.py

# copy in external files
mkdir docs/templates
cp ../templates/docs/INDEX.md docs/templates/index.md
python3 scripts/copy_templates.py

cp ../cookbook/README.md src/pages/cookbook.mdx

wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O docs/langserve.md
python3 scripts/resolve_local_links.py docs/langserve.md https://github.com/langchain-ai/langserve/tree/main/

wget -q https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md -O docs/langgraph.md
python3 scripts/resolve_local_links.py docs/langgraph.md https://github.com/langchain-ai/langgraph/tree/main/

# render
quarto render docs/
