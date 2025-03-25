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
count=$(git grep -E '(@root_validator)|(@validator)|(@field_validator)|(@pre_init)' -- "*.py" | wc -l)
# PRs that increase the current count will not be accepted.
# PRs that decrease update the code in the repository
# and allow decreasing the count of are welcome!
current_count=123

if [ "$count" -gt "$current_count" ]; then
  echo "The PR seems to be introducing new usage of @root_validator and/or @field_validator."
  echo "git grep -E '(@root_validator)|(@validator)|(@field_validator)|(@pre_init)' | wc -l returned $count"
  echo "whereas the expected count should be equal or less than $current_count"
  echo "Please update the code to instead use @model_validator or __init__"
  exit 1
elif [ "$count" -lt "$current_count" ]; then
    echo "Please update the $current_count variable in ./scripts/check_pydantic.sh to $count"
    exit 1
fi

# We do not want to be using pydantic-settings. There's already a pattern to look
# up env settings in the code base, and we want to be using the existing pattern
# rather than relying on an external dependency.
count=$(git grep -E '^import pydantic_settings|^from pydantic_settings' -- "*.py" | wc -l)

# PRs that increase the current count will not be accepted.
# PRs that decrease update the code in the repository
# and allow decreasing the count of are welcome!
current_count=8

if [ "$count" -gt "$current_count" ]; then
  echo "The PR seems to be introducing new usage pydantic_settings."
  echo "git grep -E '^import pydantic_settings|^from pydantic_settings' | wc -l returned $count"
  echo "whereas the expected count should be equal or less than $current_count"
  echo "Please update the code to use Field(default_factory=from_env(..)) or Field(default_factory=secret_from_env(..))"
  exit 1
elif [ "$count" -lt "$current_count" ]; then
    echo "Please update the $current_count variable in ./scripts/check_pydantic.sh to $count"
    exit 1
fi
