# langchain-elasticsearch

This package contains the LangChain integration with Elasticsearch.

## Installation

```bash
pip install -U langchain-elasticsearch
```

TODO document how to get id and key

## Usage

The `ElasticsearchStore` class exposes the connection to the Pinecone vector store.

```python
from langchain_elasticsearch import ElasticsearchStore

embeddings = ... # use a LangChain Embeddings class

vectorstore = ElasticsearchStore(
    es_cloud_id="your-cloud-id",
    es_api_key="your-api-key",
    index_name="your-index-name",
    embeddings=embeddings,
)
```

