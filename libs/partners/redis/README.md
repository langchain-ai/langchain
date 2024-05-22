# langchain-redis

This package contains the LangChain integration with Redis, providing powerful tools for vector storage, semantic caching, and chat history management.

## Installation

```bash
pip install -U langchain-redis
```

This will install the package along with its dependencies, including `redis`, `redisvl`, and `ulid`.

## Configuration

To use this package, you need to have a Redis instance running. You can configure the connection by setting the following environment variable:

```bash
export REDIS_URL="redis://username:password@localhost:6379"
```

Alternatively, you can pass the Redis URL directly when initializing the components or use the `RedisConfig` class for more detailed configuration.

## Features

### 1. Vector Store

The `RedisVectorStore` class provides a vector database implementation using Redis.

#### Usage

```python
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_core.embeddings import Embeddings

embeddings = Embeddings()  # Your preferred embedding model

config = RedisConfig(
    index_name="my_vectors",
    redis_url="redis://localhost:6379",
    distance_metric="COSINE"  # Options: COSINE, L2, IP
)

vector_store = RedisVectorStore(embeddings, config=config)

# Adding documents
texts = ["Document 1 content", "Document 2 content"]
metadatas = [{"source": "file1"}, {"source": "file2"}]
vector_store.add_texts(texts, metadatas=metadatas)

# Adding documents with custom keys
custom_keys = ["doc1", "doc2"]
vector_store.add_texts(texts, metadatas=metadatas, keys=custom_keys)

# Similarity search
query = "Sample query"
docs = vector_store.similarity_search(query, k=2)

# Similarity search with score
docs_and_scores = vector_store.similarity_search_with_score(query, k=2)

# Similarity search with filtering
filter_expr = Tag("category") == "science"
filtered_docs = vector_store.similarity_search(query, k=2, filter=filter_expr)

# Maximum marginal relevance search
docs = vector_store.max_marginal_relevance_search(query, k=2, fetch_k=10)
```

#### Features
- Efficient vector storage and retrieval
- Support for metadata filtering
- Multiple distance metrics: Cosine similarity, L2, and Inner Product
- Maximum marginal relevance search
- Custom key support for document indexing

### 2. Cache

The `RedisCache` and `RedisSemanticCache` classes provide caching mechanisms for LLM calls.

#### Usage

```python
from langchain_redis import RedisCache, RedisSemanticCache
from langchain_core.language_models import LLM
from langchain_core.embeddings import Embeddings

# Standard cache
cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)

# Semantic cache
embeddings = Embeddings()  # Your preferred embedding model
semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embeddings,
    distance_threshold=0.1
)

# Using cache with an LLM
llm = LLM(cache=cache)  # or LLM(cache=semantic_cache)

# Async cache operations
await cache.aupdate("prompt", "llm_string", [Generation(text="cached_response")])
cached_result = await cache.alookup("prompt", "llm_string")
```

#### Features
- Efficient caching of LLM responses
- TTL support for automatic cache expiration
- Semantic caching for similarity-based retrieval
- Asynchronous cache operations

### 3. Chat History

The `RedisChatMessageHistory` class provides a Redis-based storage for chat message history.

#### Usage

```python
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

history = RedisChatMessageHistory(
    session_id="user_123",
    redis_url="redis://localhost:6379",
    ttl=3600  # Optional: set TTL for message expiration
)

# Adding messages
history.add_user_message("Hello, AI!")
history.add_ai_message("Hello, human! How can I assist you today?")
history.add_message(SystemMessage(content="This is a system message"))

# Retrieving messages
messages = history.messages

# Searching messages
results = history.search_messages("assist")

# Get the number of messages
message_count = len(history)

# Clear history
history.clear()
```

#### Features
- Persistent storage of chat messages
- Support for different message types (Human, AI, System)
- Message searching capabilities
- Automatic expiration with TTL support
- Message count functionality

## Advanced Configuration

The `RedisConfig` class allows for detailed configuration of the Redis integration:

```python
from langchain_redis import RedisConfig

config = RedisConfig(
    index_name="my_index",
    redis_url="redis://localhost:6379",
    distance_metric="COSINE",
    key_prefix="my_prefix",
    vector_datatype="FLOAT32",
    storage_type="hash",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "price", "type": "numeric"}
    ]
)
```

Refer to the inline documentation for detailed information on these configuration options.

## Error Handling and Logging

The package uses Python's standard logging module. You can configure logging to get more information about the package's operations:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Error handling is done through custom exceptions. Make sure to handle these exceptions in your application code.

## Performance Considerations

- For large datasets, consider using batched operations when adding documents to the vector store.
- Adjust the `k` and `fetch_k` parameters in similarity searches to balance between accuracy and performance.
- Use appropriate indexing algorithms (FLAT, HNSW) based on your dataset size and query requirements.

## Examples

For more detailed examples and use cases, please refer to the `docs/` directory in this repository.

## Contributing / Development

### Unit Tests

To install dependencies for unit tests:

```bash
poetry install --with test
```

To run unit tests:

```bash
make test
```

To run a specific test:

```bash
TEST_FILE=tests/unit_tests/test_imports.py make test
```

## Integration Tests

You would need an OpenAI API Key to run the integration tests:

```bash
export OPENAI_API_KEY=sk-J3nnYJ3nnYWh0Can1Turnt0Ug1VeMe50mth1n1cAnH0ld0n2`
```

To install dependencies for integration tests:

```bash
poetry install --with test,test_integration
```

To run integration tests:

```bash
make integration_tests
```

## License

This project is licensed under the MIT License (LICENSE).