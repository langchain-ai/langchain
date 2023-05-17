# LangChain Plus Client

The `LangChainPlusClient` is useful for interacting with a tracing server. 
This guide explains how to make the client, how to connect to the server, and some of functionality it enables.
For more information on using the client to evaluate agents on Datasets, check out our [Datasets](./datasets.md) guide.

This guide assumes you already have a [hosted account](../tracing/hosted_installation.md) or are running the
 [locally hosted tracing server](../tracing/local_installation.md).

## Installation

The `LangChainPlusClient` is included as a part of your `langchain` installation. To install or upgrade, run:

```bash
pip install -U langchain
```


## Creating LangChainPlusClient

The `LangChainPlusClient` connects to the tracing server's REST API. To create a client:

```python
# import os
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"  # Uncomment this line if you want to use the hosted version
# os.environ["LANGCHAIN_API_KEY"] = "<YOUR-LANGCHAINPLUS-API-KEY>"  # Uncomment this line if you want to use the hosted version.

from langchain.client import LangChainPlusClient

client = LangChainPlusClient()
```

## Listing Sessions

You can easily interact with Runs, Sessions (groups of traced runs), and Datasets with the client.
For instance, to retrieve all of the top level runs in the default session, run:

```python
session_name = "default"
runs = client.list_runs(session_name=session_name)
```

To list sessions:

```python
sessions = client.list_sessions()
```


## Datasets

The client is also useful for evaluating agents and LLMs on datasets. Check out the [Datasets](./datasets.md) guide for more information.

