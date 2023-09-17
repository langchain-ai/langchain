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

git grep '^from langchain' langchain/adapters | grep -vE 'from langchain.(pydantic_v1|utils|schema|load)' && exit 1 || exit 0

git grep '^from langchain' langchain/callbacks | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env)' && exit 1 || exit 0

git grep '^from langchain' langchain/storage | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|storage)' && exit 1 || exit 0

git grep '^from langchain' langchain/prompts | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|_api)' && exit 1 || exit 0

git grep '^from langchain' langchain/output_parsers | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|_api|output_parsers)' && exit 1 || exit 0

git grep '^from langchain' langchain/utilities | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|utilities)' && exit 1 || exit 0

git grep '^from langchain' langchain/llms | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|llms)' && exit 1 || exit 0

git grep '^from langchain' langchain/chat_models | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|llms|prompts|adapters|chat_models)' && exit 1 || exit 0

git grep '^from langchain' langchain/embeddings | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|storage|llms|embeddings)' && exit 1 || exit 0

git grep '^from langchain' langchain/docstore | grep -vE 'from langchain.(pydantic_v1|utils|schema|docstore)' && exit 1 || exit 0

git grep '^from langchain' langchain/vectorstores | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|_api|storage|llms|docstore|vectorstores)' && exit 1 || exit 0