#!/bin/bash

# Initialize a variable to keep track of errors
errors=0

# Check the conditions
git grep 'from langchain import' langchain | grep -vE 'from langchain import (__version__|hub)' && errors=$((errors+1))
git grep 'from langchain ' langchain/pydantic_v1 && errors=$((errors+1))
git grep 'from langchain' langchain/load | grep -vE 'from langchain.(pydantic_v1|load)' && errors=$((errors+1))
git grep 'from langchain' langchain/utils | grep -vE 'from langchain.(pydantic_v1|utils)' && errors=$((errors+1))

# Decide on an exit status based on the errors
if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
