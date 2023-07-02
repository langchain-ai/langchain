# Cheat sheet

Build with the base dev image plus my snowflake dependencies

```
docker-compose -f docker-compose.yaml -f compose.local.yaml build
```

Start container using azure LLM or openai LLM and the run test

## Azure LLM

```
docker-compose -f docker-compose.yaml -f compose.local.yaml -f compose.azure.yaml up
docker exec -it langchain_azure pytest --pdb -s ../tests/integration_tests/chains/test_cpal.py
```

## OpenAI LLM for Jupyter notebook development

```
docker-compose -f docker-compose.yaml -f compose.local.yaml -f compose.openai.yaml up
docker exec -it langchain_openai pytest --pdb -s ../tests/integration_tests/chains/test_cpal.py
```


## Patch `pyproject.toml` 

**Problem:** `poetry install` timeout error on MS URL to `azure-sdk-dev` in the `pyproject.toml`


**Solution:** comment out lines with `azure-sdk-dev` in the `pyproject.toml`

or run this `patch` command

```
cd ~/workspace/langchain
patch pyproject.toml ~/workspace/my-langchain-tooling/pyproject.toml.diff
```

Here is the `pyproject.toml.diff` patch rendered in color!

```diff
110c110
< azure-search-documents = {version = "11.4.0a20230509004", source = "azure-sdk-dev", optional = true}
---
> # azure-search-documents = {version = "11.4.0a20230509004", source = "azure-sdk-dev", optional = true}
337,340c337,340
< [[tool.poetry.source]]
< name = "azure-sdk-dev"
< url = "https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/"
< secondary = true
---
> #[[tool.poetry.source]]
> #name = "azure-sdk-dev"
> #url = "https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/"
> #secondary = true
```


## Point `pyright` language server to container's packages

#### Problem

`pyright` checks the laptop's python packages, rather than the
container's python packages. Because of this my neovim editor displays noisy
hints.

```
    3 from typing import Optional, Any  # , Union
E   4 import networkx as nx     ■ Import "networkx" could not be resolved
E   5 import pandas as pd     ■ Import "pandas" could not be resolved
E   6 from pydantic import (     ■ Import "pydantic" could not be resolved
    7     BaseModel,
    8     Field,
    9     validator,
   10     root_validator,
```


#### Solution

##### pyright config

```
ln -s /Users/borisdev/workspace/my-langchain-tooling/pyrightconfig.json /Users/borisdev/workspace/langchain/pyrightconfig.json
```

##### Cuda
```
docker exec -it langchain bash
pip install -E ### BREAKS WITH MAC M1 no CUDA
```

## CPAL Ideas Backlog

### CPAL code enhancements

- [ ] include DB table schema context with SQL prompt
- [ ] multivariate
- [ ] PAL-to-LLM feedback. Send failed unit test results back to LLM to give better code
- [ ] from list to stack of operations 
- [ ] Nested causal models: Each entity can depend on an entire causal system (recursion)
- [ ] acyclical (carry over after leaves, feedback, indirect effects)
- [ ] add time reference `x[t+1] = x[t] + y[t]`
- [ ] report both ancestors and descendants 

### CPAL potential applications

- Plan as Code
- Simulate crypto trades under diff block conditions

## Github Codespace 

### Why?

- Live interactive demos
- Code from any browser 

### How?

Use my terminal (`zsh`) and editor `neovim` setup 

Move these two files into my the dir of my langchain clone.

- `.devcontainer/devcontainer.json`
- `.devcontainer/bootstrap.sh`

[Trouble shooting codespaces](https://docs.github.com/en/codespaces/troubleshooting/troubleshooting-personalization-for-codespaces)

### Github CLI

Optional: ssh into codespace with [github cli](https://github.com/cli/cli#installation) on mac to ssh into container

```console
brew install gh
gh codespace list
gh codespace list | head -n1
gh codespace ssh -c borisdev-improved-memory-4v4j5xx972jp7r
gh codespace list | head -n1 | cut -f1 | xargs -I {} gh codespace ssh -c {}
pytest -s tests/integration_tests/chains/test_pal.py
```
