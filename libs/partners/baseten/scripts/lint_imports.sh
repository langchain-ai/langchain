#!/bin/bash

set -eu

# Navigate to the directory of this script
cd "$(dirname "$0")/.."

# Run import checks
python scripts/check_imports.py $(find langchain_baseten -name "*.py")
