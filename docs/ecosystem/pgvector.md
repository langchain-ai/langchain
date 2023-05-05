# PGVector

This page covers how to use the Postgres [PGVector](https://github.com/pgvector/pgvector) ecosystem within LangChain
It is broken into two parts: installation and setup, and then references to specific PGVector wrappers.

## Installation
- Install the Python package with `pip install pgvector`


## Setup
1. The first step is to create a database with the `pgvector` extension installed.

    Follow the steps at [PGVector Installation Steps](https://github.com/pgvector/pgvector#installation) to install the database and the extension. The docker image is the easiest way to get started.

## Wrappers

### VectorStore

There exists a wrapper around Postgres vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores.pgvector import PGVector
```

PGVector embedding size is not autodetected. If you are using ChatGPT or any other embedding with 1536 dimensions
default is fine. If you are going to use for example HuggingFaceEmbeddings you need to set the environment variable named `PGVECTOR_VECTOR_SIZE`
to the needed value, In case of HuggingFaceEmbeddings is would be: `PGVECTOR_VECTOR_SIZE=768`

### Usage

For a more detailed walkthrough of the PGVector Wrapper, see [this notebook](../modules/indexes/vectorstores/examples/pgvector.ipynb)
