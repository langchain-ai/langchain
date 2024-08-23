#!/bin/bash

set -e

make install-vercel-deps

make build

rm -rf docs
mv build/output-new/docs ./

mkdir static/api_reference

git clone --depth=1 https://github.com/baskaryan/langchain-api-docs-build.git

mv -r langchain-api-docs-build/api_reference_build/html/* static/api_reference/

