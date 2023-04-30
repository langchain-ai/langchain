# SearxNG Search API

This page covers how to use the SearxNG search API within LangChain.
It is broken into two parts: installation and setup, and then references to the specific SearxNG API wrapper.

## Installation and Setup

While it is possible to utilize the wrapper in conjunction with  [public searx
instances](https://searx.space/) these instances frequently do not permit API
access (see note on output format below) and have limitations on the frequency
of requests. It is recommended to opt for a self-hosted instance instead.

### Self Hosted Instance:

See [this page](https://searxng.github.io/searxng/admin/installation.html) for installation instructions.

When you install SearxNG, the only active output format by default is the HTML format.
You need to activate the `json` format to use the API. This can be done by adding the following line to the `settings.yml` file:
```yaml
search:
    formats:
        - html
        - json
```
You can make sure that the API is working by issuing a curl request to the API endpoint:

`curl -kLX GET --data-urlencode q='langchain' -d format=json http://localhost:8888`

This should return a JSON object with the results.


## Wrappers

### Utility

To use the wrapper we need to pass the host of the SearxNG instance to the wrapper with:
    1. the named parameter `searx_host` when creating the instance.
    2. exporting the environment variable `SEARXNG_HOST`.

You can use the wrapper to get results from a SearxNG instance. 

```python
from langchain.utilities import SearxSearchWrapper
s = SearxSearchWrapper(searx_host="http://localhost:8888")
s.run("what is a large language model?")
```

### Tool

You can also load this wrapper as a Tool (to use with an Agent).

You can do this with:

```python
from langchain.agents import load_tools
tools = load_tools(["searx-search"],
                    searx_host="http://localhost:8888",
                    engines=["github"])
```

Note that we could _optionally_ pass custom engines to use.

If you want to obtain results with metadata as *json* you can use:
```python
tools = load_tools(["searx-search-results-json"],
                    searx_host="http://localhost:8888",
                    num_results=5)
```

For more information on tools, see [this page](../modules/agents/tools/getting_started.md)
