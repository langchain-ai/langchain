# langchain-baseten

This package contains the LangChain integration with Baseten.

## Installation

```bash
pip install langchain-baseten
```

The embeddings functionality uses Baseten's Performance Client for optimized performance:

```bash
pip install baseten-performance-client
```

## Chat Models

`ChatBaseten` class exposes chat models from Baseten.

```python
from langchain_baseten import ChatBaseten

# Initialize the chat model
chat = ChatBaseten(
    model="deepseek-ai/DeepSeek-V3-0324",  # Replace with your model
    baseten_api_key="your-api-key",  # Or set BASETEN_API_KEY env var
)

# Use the chat model
response = chat.invoke("Hello, how are you?")
print(response.content)
```

## Embeddings

`BasetenEmbeddings` class exposes embedding models from Baseten.

```python
from langchain_baseten import BasetenEmbeddings

# Initialize the embeddings model
embeddings = BasetenEmbeddings(
    model="my_embedding_model",  # Replace with your model name
    model_url="https://model-<id>.api.baseten.co/environments/production/sync",  # Your model URL
    baseten_api_key="your-api-key",  # Or set BASETEN_API_KEY env var
)

# Embed documents
vectors = embeddings.embed_documents(["Hello world", "How are you?"])
print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")

# Embed a single query
query_vector = embeddings.embed_query("What is the meaning of life?")
print(f"Query embedding dimension: {len(query_vector)}")
```

## Configuration

You can configure the Baseten integration using environment variables:

- `BASETEN_API_KEY`: Your Baseten API key

## Supported Models

Baseten supports various models through their OpenAI-compatible API. You can use any model slug available in your Baseten account.

For more information about available models, visit the [Baseten documentation](https://docs.baseten.co/).
