#!/bin/bash
#
# This checks for usage of pickle in the package.
#
# Usage: ./scripts/check_pickle.sh /path/to/repository
#
# Check if a path argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/repository"
  exit 1
fi

repository_path="$1"

# Search for lines matching the pattern within the specified repository
result=$(git -C "$repository_path" grep -E 'pickle.load\(|pickle.loads\(' | grep -v '# ignore\[pickle\]: explicit-opt-in')

# Check if any matching lines were found
if [ -n "$result" ]; then
  echo "ERROR: The following lines need to be updated:"
  echo "$result"
  echo "Please avoid using pickle or cloudpickle."  
  echo "If you must, then add:"
  echo "1. A security notice (scan the code for examples)"
  echo "2. Code path should be opt-in."
  exit 1
fi
