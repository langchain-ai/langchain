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
result=$(git -C "$repository_path" grep -En '^import pydantic|^from pydantic')

# Check if any matching lines were found
if [ -n "$result" ]; then
  echo "ERROR: The following lines need to be updated:"
  echo "$result"
  echo "Please replace the code with an import from langchain_core.pydantic_v1."
  echo "For example, replace 'from pydantic import BaseModel'"
  echo "with 'from langchain_core.pydantic_v1 import BaseModel'"
  exit 1
fi

# Forbid vanilla usage of @root_validator
# This prevents the code from using either @root_validator or @root_validator()
# Search for lines matching the pattern within the specified repository
result=$(git -C "$repository_path" grep -En '(@root_validator\s*$)|(@root_validator\(\))' -- '*.py')

# Check if any matching lines were found
if [ -n "$result" ]; then
  echo "ERROR: The following lines need to be updated:"
  echo
  echo "$result"
  echo
  echo "Please replace @root_validator or @root_validator() with either:"
  echo
  echo "@root_validator(pre=True) or @root_validator(pre=False, skip_on_failure=True)"
  exit 1
fi
