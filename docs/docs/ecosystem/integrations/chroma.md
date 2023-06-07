# Chroma

>[Chroma](https://docs.trychroma.com/getting-started) is a database for building AI applications with embeddings.

## Installation and Setup

```bash
pip install chromadb
```


## VectorStore

There exists a wrapper around Chroma vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

```python
from langchain.vectorstores import Chroma
```

For a more detailed walkthrough of the Chroma wrapper, see [this notebook](../modules/indexes/vectorstores/getting_started.ipynb)

## Retriever

See a [usage example](../modules/indexes/retrievers/examples/chroma_self_query.ipynb).

```python
from langchain.retrievers import SelfQueryRetriever
```
