#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "${SCRIPT_DIR}"

mkdir -p ../_dist
cp -r . ../_dist
cd ../_dist
poetry run python scripts/model_feat_table.py
poetry run nbdoc_build --srcdir docs
cp ../cookbook/README.md src/pages/cookbook.mdx
cp ../.github/CONTRIBUTING.md docs/contributing.md
wget https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O docs/guides/deployments/langserve.md
poetry run python scripts/generate_api_reference_links.py
yarn install
yarn start
