#!/bin/bash

set -eu

# Initialize a variable to keep track of errors
errors=0

# make sure not importing from langchain_glm
git --no-pager grep '^from langchain_glm\.' . && errors=$((errors+1))

# Decide on an exit status based on the errors
if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
