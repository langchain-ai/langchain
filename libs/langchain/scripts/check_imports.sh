#!/bin/bash

set -eu

git grep '^from langchain import' langchain | grep -vE 'from langchain import (__version__|hub)' && exit 1 || exit 0

# Pydantic bridge should not import from any other module
git grep '^from langchain ' langchain/pydantic_v1 && exit 1 || exit 0

# load should not import from anything except itself and pydantic_v1
git grep '^from langchain' langchain/load | grep -vE 'from langchain.(pydantic_v1)' && exit 1 || exit 0

# utils should not import from anything except itself and pydantic_v1
git grep '^from langchain' langchain/utils | grep -vE 'from langchain.(pydantic_v1|utils)' && exit 1 || exit 0

git grep '^from langchain' langchain/schema | grep -vE 'from langchain.(pydantic_v1|utils|schema|load)' && exit 1 || exit 0

git grep '^from langchain' langchain/callbacks | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env)'

git grep '^from langchain' langchain/prompts | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|_api)'

git grep '^from langchain' langchain/output_parsers | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|_api|output_parsers)'

git grep '^from langchain' langchain/utilities | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|utilities)'

