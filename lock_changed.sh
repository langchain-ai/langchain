#! /bin/sh
git diff --name-only --diff-filter=d $(ARGS) | grep -E '\pyproject.toml$$' | xargs dirname