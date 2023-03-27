# Wolfram Alpha Wrapper

This page covers how to use the Wolfram Alpha API within LangChain.
It is broken into two parts: installation and setup, and then references to specific Wolfram Alpha wrappers.

## Installation and Setup
- Install requirements with `pip install wolframalpha`
- Go to wolfram alpha and sign up for a developer account [here](https://developer.wolframalpha.com/)
- Create an app and get your APP ID
- Set your APP ID as an environment variable `WOLFRAM_ALPHA_APPID`


## Wrappers

### Utility

There exists a WolframAlphaAPIWrapper utility which wraps this API. To import this utility:

```python
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
```

For a more detailed walkthrough of this wrapper, see [this notebook](../modules/agents/tools/examples/wolfram_alpha.ipynb).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
```python
from langchain.agents import load_tools
tools = load_tools(["wolfram-alpha"])
```

For more information on this, see [this page](../modules/agents/tools/getting_started.md)
