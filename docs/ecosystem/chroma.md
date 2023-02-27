# Chroma

This page covers how to use the Chroma ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Chroma wrappers.

## Installation and Setup
- Install the Python package with `pip install chromadb`
## Wrappers

### VectorStore

There exists a wrapper around Chroma vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores import Chroma
```

For a more detailed walkthrough of the Chroma wrapper, see [this notebook](../modules/indexes/examples/vectorstores.ipynb)
