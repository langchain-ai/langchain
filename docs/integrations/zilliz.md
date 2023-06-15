# Zilliz

>[Zilliz Cloud](https://zilliz.com/doc/quick_start) is a fully managed service on cloud for `LF AI Milvus®`,


## Installation and Setup

Install the Python SDK:
```bash
pip install pymilvus
```

## Vectorstore

A wrapper around Zilliz indexes allows you to use it as a vectorstore,
whether for semantic search or example selection.

```python
from langchain.vectorstores import Milvus
```

For a more detailed walkthrough of the Miluvs wrapper, see [this notebook](../modules/indexes/vectorstores/examples/zilliz.ipynb)
