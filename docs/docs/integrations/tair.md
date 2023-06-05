# Tair

This page covers how to use the Tair ecosystem within LangChain.

## Installation and Setup

Install Tair Python SDK with `pip install tair`.

## Wrappers

### VectorStore

There exists a wrapper around TairVector, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:

```python
from langchain.vectorstores import Tair
```

For a more detailed walkthrough of the Tair wrapper, see [this notebook](../modules/indexes/vectorstores/examples/tair.ipynb)
