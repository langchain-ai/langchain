# OpenWeatherMap API

This page covers how to use the OpenWeatherMap API within LangChain.
It is broken into two parts: installation and setup, and then references to specific OpenWeatherMap API wrappers.

## Installation and Setup

- Install requirements with `pip install pyowm`
- Go to OpenWeatherMap and sign up for an account to get your API key [here](https://openweathermap.org/api/)
- Set your API key as `OPENWEATHERMAP_API_KEY` environment variable

## Wrappers

### Utility

There exists a OpenWeatherMapAPIWrapper utility which wraps this API. To import this utility:

```python
from langchain.utilities.openweathermap import OpenWeatherMapAPIWrapper
```

For a more detailed walkthrough of this wrapper, see [this notebook](../modules/agents/tools/examples/openweathermap.ipynb).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:

```python
from langchain.agents import load_tools
tools = load_tools(["openweathermap-api"])
```

For more information on this, see [this page](../modules/agents/tools/getting_started.md)
