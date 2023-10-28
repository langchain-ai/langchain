# Elasticsearch RAG Example

Using Langserve and ElasticSearch to build a RAG search example for answering questions on workplace documents.

Relies on sentence transformer `MiniLM-L6-v2` for embedding passages and questions.

## Running Elasticsearch

There are a number of ways to run Elasticsearch.

### Elastic Cloud

Create a free trial account on [Elastic Cloud](https://cloud.elastic.co/registration?utm_source=langchain&utm_content=langserve).

Once you have created an account, you can create a deployment. With a deployment, you can use these environment variables to connect to your Elasticsearch instance:

```bash
export ELASTIC_CLOUD_ID = <ClOUD_ID>
export ELASTIC_USERNAME = <ClOUD_USERNAME>
export ELASTIC_PASSWORD = <ClOUD_PASSWORD>
```

### Docker

For local development, you can use Docker:

```bash
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.10.0
```

This will run Elasticsearch on port 9200. You can then check that it is running by visiting [http://localhost:9200](http://localhost:9200).

With a deployment, you can use these environment variables to connect to your Elasticsearch instance:

```bash
export ES_URL = "http://localhost:9200"
```

## Documents

To load fictional workplace documents, run the following command from the root of this repository:

```bash
python ./data/load_documents.py
```

However, you can choose from a large number of document loaders [here](https://python.langchain.com/docs/integrations/document_loaders).

## Installation

```bash
# from inside your LangServe instance
poe add rag-elasticsearch
```
