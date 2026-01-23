#!/bin/bash

set -eu

# make sure not importing from langchain, langchain_experimental, or langchain_community
errors=0

git --no-pager grep '^from langchain\.' . && errors=$((errors+1))
git --no-pager grep '^from langchain_community\.' . && errors=$((errors+1))
git --no-pager grep '^from langchain_experimental\.' . && errors=$((errors+1))

if [ "$errors" -gt 0 ]; then
  echo "Found $errors errors"
  exit 1
fi
