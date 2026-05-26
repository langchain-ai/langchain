#!/bin/bash

set -eu

errors=0

git --no-pager grep "^from langchain\." . | grep -v ":from langchain\.agents" | grep -v ":from langchain\.tools" && errors=$((errors+1))
git --no-pager grep "^from langchain_experimental\." . && errors=$((errors+1))

if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
