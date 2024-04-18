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
wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O docs/langserve.md
wget -q https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md -O docs/langgraph.md

# Duplicate changes to 0.2.x version
cp docs/integrations/llms/index.mdx versioned_docs/version-0.2.x/integrations/llms/
cp docs/integrations/chat/index.mdx versioned_docs/version-0.2.x/integrations/chat/
mkdir -p versioned_docs/version-0.2.x/templates
cp -r docs/templates/* versioned_docs/version-0.2.x/templates/
cp docs/langserve.md versioned_docs/version-0.2.x/
cp docs/langgraph.md versioned_docs/version-0.2.x/

yarn

poetry run python scripts/resolve_versioned_links_in_markdown.py versioned_docs/version-0.2.x/ /docs/0.2.x/

poetry run quarto preview docs
