#!/bin/bash
# This script searches for invalid imports in tracked files within a Git repository.
#
# Usage: ./scripts/lint_imports.sh /path/to/repository
set -eu

# Check if a path argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/repository"
  exit 1
fi

repository_path="$1"

# make sure not importing from langchain_experimental
result=$(git -C "$repository_path" grep -En '^import langchain_experimental|^from langchain_experimental' -- '*.py' || true)

# Check if any matching lines were found
if [ -n "$result" ]; then
  echo "ERROR: The following lines need to be updated:"
  echo "$result"
  echo "langchain_community should import from langchain_experimental."
  exit 1
fi

# make sure no one is importing from the built-in xml library
# instead defusedxml should be used to avoid getting CVEs.
# Whether the standard library actually poses a risk to users
# is very nuanced and depends on the user's environment.
# https://docs.python.org/3/library/xml.etree.elementtree.html

result=$(git -C "$repository_path" grep -En '^from xml.|^import xml$|^import xml.' | grep -vE "# OK: user-must-opt-in| # OK: trusted-source" || true)

if [ -n "$result" ]; then
  echo "ERROR: The following lines need to be updated:"
  echo "$result"
  echo "Triggering an error due to usage of the built-in xml library. "
  echo "Please see https://docs.python.org/3/library/xml.html#xml-vulnerabilities."
  echo "If this happens, there's likely code that's relying on the standard library "
  echo "to parse xml somewhere in the code path. "
  echo "Please update the code to force the user to explicitly opt-in to using the standard library or running the code. "
  echo "It should be **obvious** without reading the documentation that they are being forced to use the standard library. "
  echo "After this is done feel free to add a comment to the line with '# OK: user-must-opt-in', after the import. "
  echo "Lacking a clear opt-in mechanism is likely a security risk, and will result in rejection of the PR."
  exit 1
fi
