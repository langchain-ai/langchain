# Momento

This page covers how to use the [Momento](https://gomomento.com) ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Momento wrappers.

## Installation and Setup

- Sign up for a free account [here](https://docs.momentohq.com/getting-started) and get an auth token
- Install the Momento Python SDK with `pip install momento`

## Wrappers

### Cache

The Cache wrapper allows for [Momento](https://gomomento.com) to be used as a serverless, distributed, low-latency cache for LLM prompts and responses.

#### Standard Cache

The standard cache is the go-to use case for [Momento](https://gomomento.com) users in any environment.

Import the cache as follows:

```python
from langchain.cache import MomentoCache
```

And set up like so:

```python
from datetime import timedelta
from momento import CacheClient, Configurations, CredentialProvider
import langchain

# Instantiate the Momento client
cache_client = CacheClient(
    Configurations.Laptop.v1(),
    CredentialProvider.from_environment_variable("MOMENTO_AUTH_TOKEN"),
    default_ttl=timedelta(days=1))

# Choose a Momento cache name of your choice
cache_name = "langchain"

# Instantiate the LLM cache
langchain.llm_cache = MomentoCache(cache_client, cache_name)
```

### Memory

Momento can be used as a distributed memory store for LLMs.

#### Chat Message History Memory

See [this notebook](../modules/memory/examples/momento_chat_message_history.ipynb) for a walkthrough of how to use Momento as a memory store for chat message history.
