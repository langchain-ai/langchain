# langchain-chroma

This package contains the LangChain integration with Chroma.

## Installation

```bash
pip install -U langchain-chroma
```

## Usage

The `ChromaVectorStore` class exposes the connection to the Chroma vector store.

```python
from langchain_chroma import ChromaVectorStore

embeddings = ... # use a LangChain Embeddings class

vectorstore = ChromaVectorStore(embeddings=embeddings)
```
