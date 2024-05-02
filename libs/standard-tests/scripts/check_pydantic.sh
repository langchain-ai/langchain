#!/bin/bash
#
# This script searches for lines starting with "import pydantic" or "from pydantic"
# in tracked files within a Git repository.
#
# Usage: ./scripts/check_pydantic.sh /path/to/repository

# Check if a path argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/repository"
  exit 1
fi

repository_path="$1"

# Search for lines matching the pattern within the specified repository
result=$(
  git -C "$repository_path" grep -E '^[[:space:]]*import pydantic|^[[:space:]]*from pydantic' \
  -- ':!langchain_core/pydantic_*' ':!langchain_core/utils' | grep -v 'pydantic: ignore'
)

# Check if any matching lines were found
if [ -n "$result" ]; then
  echo "ERROR: The following lines need to be updated:"
  echo "$result"
  echo "Please replace the code with an import from langchain_core.pydantic_v1."
  echo "For example, replace 'from pydantic import BaseModel'"
  echo "with 'from langchain_core.pydantic_v1 import BaseModel'"
  echo "If this was intentional, you can add #  pydantic: ignore after the import to ignore this error."
  exit 1
fi
