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

# Check that we are not using features that cannot be captured via init.
# pre-init is a custom decorator that we introduced to capture the same semantics
# as @root_validator(pre=False, skip_on_failure=False) available in pydantic 1.
count=$(git grep -E '(@root_validator)|(@validator)|(@pre_init)' -- "*.py" | wc -l)
# PRs that increase the current count will not be accepted.
# PRs that decrease update the code in the repository
# and allow decreasing the count of are welcome!
current_count=337

if [ "$count" -gt "$current_count" ]; then
  echo "The PR seems to be introducing new usage of @root_validator and/or @field_validator."
  echo "git grep -E '(@root_validator)|(@validator)' | wc -l returned $count"
  echo "whereas the expected count should be equal or less than $current_count"
  echo "Please update the code to instead use __init__"
  echo "For examples, please see: "
  echo "https://gist.github.com/eyurtsev/d1dcba10c2f35626e302f1b98a0f5a3c "
  echo "This linter is here to make sure that its easier to upgrade pydantic in the future."
  exit 1
elif [ "$count" -lt "$current_count" ]; then
    echo "Please update the $current_count variable in ./scripts/check_pydantic.sh to $count"
    exit 1
fi


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
result=$(git -C "$repository_path" grep -En '(@root_validator\s*$)|(@root_validator\(\)|@root_validator\(pre=False\))' -- '*.py')

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
