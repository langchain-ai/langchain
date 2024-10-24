#!/bin/bash

# Get the working directory from the input argument, default to 'all' if not provided
WORKING_DIRECTORY=${1:-all}

# Function to delete cassettes
delete_cassettes() {
  local dir=$1
  if [ "$dir" == "all" ]; then
    echo "Deleting all cassettes..."
    rm -f docs/cassettes/*.msgpack.zlib
  else
    # Extract the filename from the directory path
    local filename=$(basename "$dir" .ipynb)
    echo "Deleting cassettes for $filename..."
    rm -f docs/cassettes/${filename}_*.msgpack.zlib
  fi
}

# Delete existing cassettes
delete_cassettes "$WORKING_DIRECTORY"

# Pre-download tiktoken files
echo "Pre-downloading tiktoken files..."
poetry run python docs/scripts/download_tiktoken.py

# Prepare notebooks
echo "Preparing notebooks for CI..."
poetry run python docs/scripts/prepare_notebooks_for_ci.py --comment-install-cells --working-directory "$WORKING_DIRECTORY"

# Run notebooks
echo "Running notebooks..."
./docs/scripts/execute_notebooks.sh "$WORKING_DIRECTORY"
