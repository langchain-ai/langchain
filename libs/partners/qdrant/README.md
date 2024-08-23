# langchain-qdrant

This package contains the LangChain integration with [Qdrant](https://qdrant.tech/).

## Installation

```bash
pip install -U langchain-qdrant
```

## Usage

The `Qdrant` class exposes the connection to the Qdrant vector store.

```python
from langchain_qdrant import Qdrant

embeddings = ... # use a LangChain Embeddings class

vectorstore = Qdrant.from_existing_collection(
    embeddings=embeddings,
    collection_name="<COLLECTION_NAME>",
    url="http://localhost:6333",
)
```
