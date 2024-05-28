#!/bin/bash

set -eu

# Initialize a variable to keep track of errors
errors=0

# make sure not importing from langchain or langchain_experimental
git --no-pager grep '^from langchain_experimental\.' . && errors=$((errors+1))

# make sure no one is importing from the built-in xml library
# instead defusedxml should be used to avoid getting CVEs.
# Whether the standary library actually poses a risk to users
# is very nuanced and dependns on user's environment.
# https://docs.python.org/3/library/xml.etree.elementtree.html
git --no-pager grep '^from xml\.' . | grep -vE "# OK: user-must-opt-in" && errors=$((errors+1))
git --no-pager grep '^import xml\.' . | grep -vE "# OK: user-must-opt-in" && errors=$((errors+1))

# Decide on an exit status based on the errors
if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
