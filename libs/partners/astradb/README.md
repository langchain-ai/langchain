# langchain-astradb

This package contains the LangChain integrations for using DataStax Astra DB.

> DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Apache Cassandra® and made conveniently available
> through an easy-to-use JSON API.

_**Note.** For a short transitional period, only some of the Astra DB integration classes are contained in this package (the remaining ones being still in `langchain-community`). In a short while, and surely by version 0.2 of LangChain, all of the Astra DB support will be removed from `langchain-community` and included in this package._

## Installation and Setup

Installation of this partner package:

```bash
pip install langchain-astradb
```

## Integrations overview

### Vector Store

```python
from langchain_astradb.vectorstores import AstraDBVectorStore

my_store = AstraDBVectorStore(
  embedding=my_embeddings,
  collection_name="my_store",
  api_endpoint="https://...",
  token="AstraCS:...",
)
```

### Store

```python
from langchain_astradb import AstraDBStore
store = AstraDBStore(
    collection_name="my_kv_store",
    api_endpoint="...",
    token="..."
)
```

### Byte Store

```python
from langchain_astradb import AstraDBByteStore
store = AstraDBByteStore(
    collection_name="my_kv_store",
    api_endpoint="...",
    token="..."
)
```

## Reference

See the [LangChain docs page](https://python.langchain.com/docs/integrations/providers/astradb) for a more detailed listing.
