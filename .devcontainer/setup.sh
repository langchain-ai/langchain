#!/bin/bash
poetry completions bash >> ~/.bash_completion
cd libs/langchain
# Installs main, test, codespell, lint, typing (all non-optional) and dev dependencies
poetry install --no-interaction --no-ansi --with dev