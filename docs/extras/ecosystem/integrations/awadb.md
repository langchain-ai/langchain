# AwaDB

>[AwaDB](https://github.com/awa-ai/awadb) is an AI Native database for the search and storage of embedding vectors used by LLM Applications.

## Installation and Setup

```bash
pip install awadb
```


## VectorStore

There exists a wrapper around AwaDB vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

```python
from langchain.vectorstores import AwaDB
```

For a more detailed walkthrough of the AwaDB wrapper, see [here](/docs/modules/data_connection/vectorstores/integrations/awadb.html).
