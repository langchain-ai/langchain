# Zilliz

This page covers how to use the Zilliz Cloud ecosystem within LangChain.
Zilliz uses the Milvus integration. 
It is broken into two parts: installation and setup, and then references to specific Milvus wrappers.

## Installation and Setup
- Install the Python SDK with `pip install pymilvus`
## Wrappers

### VectorStore

There exists a wrapper around Zilliz indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores import Milvus
```

For a more detailed walkthrough of the Miluvs wrapper, see [this notebook](../modules/indexes/vectorstores/examples/zilliz.ipynb)
