# langchain-pinecone

This package contains the LangChain integration with Pinecone.

## Installation

```bash
pip install -U langchain-pinecone
```

And you should configure credentials by setting the following environment variables:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_ENVIRONMENT`

## Usage

The `Pinecone` class exposes the connection to the Pinecone vector store.

```python
from langchain_pinecone import Pinecone

embeddings = ... # use a LangChain Embeddings class

vectorstore = Pinecone(embeddings=embeddings)
```
