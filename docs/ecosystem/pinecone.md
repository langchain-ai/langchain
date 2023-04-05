# Pinecone

This page covers how to use the Pinecone ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Pinecone wrappers.

## Installation and Setup
- Install the Python SDK with `pip install pinecone-client`
## Wrappers

### VectorStore

There exists a wrapper around Pinecone indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores import Pinecone
```

For a more detailed walkthrough of the Pinecone wrapper, see [this notebook](../modules/indexes/vectorstores/examples/pinecone.ipynb)
