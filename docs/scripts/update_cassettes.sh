#!/bin/bash

# Use this script to update cassettes for a notebook. The script does the following:
#
# 1. Delete existing cassettes for the specified notebook
# 2. Pre-download and cache nltk and tiktoken files
# 3. Modify the notebook to generate cassettes for each cell.
# 4. Execute the notebook.
#
# Important: make sure the notebook is in a clean state, with any desired changes
# staged or committed. The script will modify the notebook in place, and these
# modifications should be discarded after the cassettes are generated.
#
# Usage:
# In monorepo env, `poetry install --with dev,test`
# `./docs/scripts/update_cassettes.sh path/to/notebook`
# e.g., `./docs/scripts/update_cassettes.sh docs/docs/how_to/tool_choice.ipynb`
#
# Make sure to set any env vars required by the notebook.


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
echo "Pre-downloading nltk and tiktoken files..."
poetry run python docs/scripts/cache_data.py

# Prepare notebooks
echo "Preparing notebooks for CI..."
poetry run python docs/scripts/prepare_notebooks_for_ci.py --comment-install-cells --working-directory "$WORKING_DIRECTORY"

# Run notebooks
echo "Running notebooks..."
./docs/scripts/execute_notebooks.sh "$WORKING_DIRECTORY"
