#!/bin/bash

# Use sed to replace "Langchain" with "Gigachain" in lines
sed -i -E '/packages = \[/,/^\s*\]/! s/langchain/gigachain/' pyproject.toml
