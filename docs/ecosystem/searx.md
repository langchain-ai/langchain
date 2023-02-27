# SearxNG Search API

This page covers how to use the SearxNG search API within LangChain.
It is broken into two parts: installation and setup, and then references to the specific SearxNG API wrapper.

## Installation and Setup

- You can find a list of public SearxNG instances [here](https://searx.space/). 
- It recommended to use a self-hosted instance to avoid abuse on the public instances. Also note that public instances often have a limit on the number of requests.
- To run a self-hosted instance see [this page](https://searxng.github.io/searxng/admin/installation.html) for more information.
- To use the tool you need to provide the searx host url by:
    1. passing the named parameter `searx_host` when creating the instance.
    2. exporting the environment variable `SEARXNG_HOST`. 

## Wrappers

### Utility

You can use the wrapper to get results from a SearxNG instance. 

```python
from langchain.utilities import SearxSearchWrapper
```

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:

```python
from langchain.agents import load_tools
tools = load_tools(["searx-search"], searx_host="https://searx.example.com")
```

For more information on this, see [this page](../modules/agents/tools.md)
