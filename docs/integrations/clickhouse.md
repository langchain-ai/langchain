# ClickHouse

This page covers how to use ClickHouse Vector Search within LangChain.

[ClickHouse](https://clickhouse.com) is a open source real-time OLAP database with full SQL support and a wide range of functions to assist users in writing analytical queries. Some of these functions and data structures perform distance operations between vectors, enabling ClickHouse to be used as a vector database.

Due to the fully parallelized query pipeline, ClickHouse can process vector search operations very quickly, especially when performing exact matching through a linear scan over all rows, delivering processing speed comparable to dedicated vector databases.

High compression levels, tunable through custom compression codecs, enable very large datasets to be stored and queried. ClickHouse is not memory-bound, allowing multi-TB datasets containing embeddings to be queried.

The capabilities for computing the distance between two vectors are just another SQL function and can be effectively combined with more traditional SQL filtering and aggregation capabilities. This allows vectors to be stored and queried alongside metadata, and even rich text, enabling a broad array of use cases and applications.

Finally, experimental ClickHouse capabilities like [Approximate Nearest Neighbour (ANN) indices](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/annindexes) support faster approximate matching of vectors and provide a promising development aimed to further enhance the vector matching capabilities of ClickHouse.

## Installation
- Install clickhouse server by [binary](https://clickhouse.com/docs/en/install) or [docker image](https://hub.docker.com/r/clickhouse/clickhouse-server/)
- Install the Python SDK with `pip install clickhouse-connect`

### Configure clickhouse vector index

Customize `ClickhouseSettings` object with parameters

    ```python
    from langchain.vectorstores import ClickHouse, ClickhouseSettings
    config = ClickhouseSettings(host="<clickhouse-server-host>", port=8123, ...)
    index = Clickhouse(embedding_function, config)
    index.add_documents(...)
    ```
  
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

There exists a wrapper around open source Clickhouse database, allowing you to use it as a vectorstore,
whether for semantic search or similar example retrieval.

To import this vectorstore:
```python
from langchain.vectorstores import Clickhouse
```

For a more detailed walkthrough of the MyScale wrapper, see [this notebook](../modules/indexes/vectorstores/examples/clickhouse.ipynb)
