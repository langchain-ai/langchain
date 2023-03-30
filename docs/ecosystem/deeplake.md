# Deep Lake

This page covers how to use the Deep Lake ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Deep Lake wrappers. For more information.

1. Here is [whitepaper](https://www.deeplake.ai/whitepaper) and [academic paper](https://arxiv.org/pdf/2209.10785.pdf) for Deep Lake

2. Here is a set of additional resources available for review: [Deep Lake](https://github.com/activeloopai/deeplake), [Getting Started](https://docs.activeloop.ai/getting-started) and [Tutorials](https://docs.activeloop.ai/hub-tutorials)

## Installation and Setup
- Install the Python package with `pip install deeplake`

## Wrappers

### VectorStore

There exists a wrapper around Deep Lake, a data lake for Deep Learning applications, allowing you to use it as a vectorstore (for now), whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores import DeepLake
```


For a more detailed walkthrough of the Deep Lake wrapper, see [this notebook](../modules/indexes/vectorstores/examples/deeplake.ipynb)
