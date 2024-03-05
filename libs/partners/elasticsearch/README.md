# langchain-elasticsearch

This package contains the LangChain integration with Elasticsearch.

## Installation

```bash
pip install -U langchain-elasticsearch
```

## Elasticsearch setup

### Elastic Cloud

You need a running Elasticsearch deployment. The easiest way to start one is through [Elastic Cloud](https://cloud.elastic.co/).
You can sign up for a [free trial](https://www.elastic.co/cloud/cloud-trial-overview).

1. [Create a deployment](https://www.elastic.co/guide/en/cloud/current/ec-create-deployment.html)
2. Get your Cloud ID:
    1. In the [Elastic Cloud console](https://cloud.elastic.co), click "Manage" next to your deployment
    2. Copy the Cloud ID and paste it into the `es_cloud_id` parameter below
3. Create an API key:
    1. In the [Elastic Cloud console](https://cloud.elastic.co), click "Open" next to your deployment
    2. In the left-hand side menu, go to "Stack Management", then to "API Keys"
    3. Click "Create API key"
    4. Enter a name for the API key and click "Create"
    5. Copy the API key and paste it into the `es_api_key` parameter below

### Elastic Cloud

Alternatively, you can run Elasticsearch via Docker as described in the [docs](https://python.langchain.com/docs/integrations/vectorstores/elasticsearch).

## Usage

### ElasticsearchStore

The `ElasticsearchStore` class exposes Elasticsearch as a vector store.

```python
from langchain_elasticsearch import ElasticsearchStore

embeddings = ... # use a LangChain Embeddings class or ElasticsearchEmbeddings

vectorstore = ElasticsearchStore(
    es_cloud_id="your-cloud-id",
    es_api_key="your-api-key",
    index_name="your-index-name",
    embeddings=embeddings,
)
```

### ElasticsearchEmbeddings

The `ElasticsearchEmbeddings` class provides an interface to generate embeddings using a model
deployed in an Elasticsearch cluster.

```python
from langchain_elasticsearch import ElasticsearchEmbeddings

embeddings = ElasticsearchEmbeddings.from_credentials(
    model_id="your-model-id",
    input_field="your-input-field",
    es_cloud_id="your-cloud-id",
    es_api_key="your-api-key",
)
```

### ElasticsearchChatMessageHistory

The `ElasticsearchChatMessageHistory` class stores chat histories in Elasticsearch.

```python
from langchain_elasticsearch import ElasticsearchChatMessageHistory

chat_history = ElasticsearchChatMessageHistory(
    index="your-index-name",
    session_id="your-session-id",
    es_cloud_id="your-cloud-id",
    es_api_key="your-api-key",
)
```
