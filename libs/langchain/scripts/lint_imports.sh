#!/bin/bash

set -eu

# Initialize a variable to keep track of errors
errors=0

# Check the conditions
git grep '^from langchain import' langchain | grep -vE 'from langchain import (__version__|hub)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/pydantic_v1 | grep -vE 'from langchain.(pydantic_v1|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/load | grep -vE 'from langchain.(pydantic_v1|load|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/utils | grep -vE 'from langchain.(pydantic_v1|utils|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/schema | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|env|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/adapters | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/callbacks | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|_api)' && errors=$((errors+1))
# TODO: it's probably not amazing so that so many other modules depend on `langchain_community.utilities`, because there can be a lot of imports there
git grep '^from langchain\.' langchain/utilities | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|utilities|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/storage | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|storage|utilities|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/prompts | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/output_parsers | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|_api|output_parsers|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/llms | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|prompts|llms|utilities|globals|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/chat_models | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|llms|prompts|adapters|chat_models|utilities|globals|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/embeddings | grep -vE 'from langchain.(pydantic_v1|utils|schema|load|callbacks|env|storage|llms|embeddings|utilities|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/docstore | grep -vE 'from langchain.(pydantic_v1|utils|schema|docstore|_api)' && errors=$((errors+1))
git grep '^from langchain\.' langchain/vectorstores | grep -vE 'from
langchain.(pydantic_v1|utils|schema|load|callbacks|env|_api|storage|llms|docstore|vectorstores|utilities|_api)' && errors=$((errors+1))
# make sure not importing from langchain_experimental
git --no-pager grep '^from langchain_experimental\.' . && errors=$((errors+1))

# Add a basic lint rule to prevent imports from the global namespaces of langchain_community
# This lint rule won't catch imports from local scope.
# We can't add that rule without a more complex script to ignore imports from inside
# a if TYPE_CHECKING block.
git grep '^from langchain_community'  | grep -vE '# ignore: community-import' && errors=$((errors+1))

# Decide on an exit status based on the errors
if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
