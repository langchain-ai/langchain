#!/bin/bash
#
# This script searches for lines starting with "import pydantic" or "from pydantic" in tracked files
# within a Git repository. 
# 
# Usage: ./scripts/check_pydantic.sh
#

# Search for lines matching the pattern
result=$(git grep -E '^import pydantic|^from pydantic')

# Check if any matching lines were found
if [ -n "$result" ]; then
  echo "The following lines need to be updated:"
  echo "$result"
  echo "Please replace the code with an import from langchain.pydantic_v1."
  exit 1
fi
