# langchain-opensearch

A standalone package for integrating OpenSearch with LangChain. It provides `OpenSearchVectorStore` to store and query documents using vector similarity.

## Installation

```bash
pip install langchain-opensearch
```

## Usage

```python
from langchain_opensearch.vectorstores import OpenSearchVectorStore
from langchain_core.embeddings import Embeddings

class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1, 0.2]] * len(texts)

    def embed_query(self, text):
        return [0.1, 0.2]

embeddings = DummyEmbeddings()
vector_store = OpenSearchVectorStore(
    opensearch_url="http://localhost:9200",
    index_name="my_index",
    embedding_function=embeddings
)

texts = ["Hello, world!", "LangChain is awesome"]
vector_store.add_texts(texts)

results = vector_store.similarity_search("world", k=1)
print([doc.page_content for doc in results])
```

## Setting Up OpenSearch (for local testing)

Start an OpenSearch container using Docker:

```bash
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=StrongPass123!@#" \
  opensearchproject/opensearch:latest
```

Verify it's running:

```bash
curl -k -u admin:StrongPass123!@# https://localhost:9200
```

To stop and remove the container:

```bash
docker stop opensearch
docker rm opensearch
```

## Testing

### Mock Tests

```bash
python -m unittest langchain-opensearch/tests/test_vectorstores.py
```

### Real OpenSearch Test

```bash
python langchain-opensearch/test_real_opensearch.py
```
```