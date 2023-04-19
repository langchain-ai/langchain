# MyScale

This page covers how to use the MyScale ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific MyScale wrappers.

## Installation and Setup
- Install the Python SDK with `pip install clickhouse-connect`
  
## Wrappers
supported functions:
- `add_texts`
- `add_documents`
- `from_texts`
- `from_documents`
- `similarity_search`
- `asimilarity_search`
- `similarity_search_by_vector`
- `asimilarity_search_by_vector`
- `similarity_search_with_relevance_scores`

### VectorStore

There exists a wrapper around MyScale database, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores import MyScale
```

For a more detailed walkthrough of the MyScale wrapper, see [this notebook](../modules/indexes/vectorstores/examples/myscale.ipynb)
