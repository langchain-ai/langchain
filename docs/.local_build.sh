#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "${SCRIPT_DIR}"

mkdir -p ../_dist
rsync -ruv --exclude node_modules --exclude api_reference --exclude .venv --exclude .docusaurus . ../_dist
cd ../_dist
poetry run python scripts/model_feat_table.py
cp ../cookbook/README.md src/pages/cookbook.mdx
mkdir -p docs/templates
cp ../templates/docs/INDEX.md docs/templates/index.md
poetry run python scripts/copy_templates.py
wget https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O docs/langserve.md

yarn

poetry run quarto preview docs
