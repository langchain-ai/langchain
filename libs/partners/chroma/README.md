# langchain-chroma

This package contains the LangChain integration with Chroma.

## Installation

```bash
pip install -U langchain-chroma
```

## Usage

The `Chroma` class exposes the connection to the Chroma vector store.

```python
from langchain_chroma import Chroma

embeddings = ... # use a LangChain Embeddings class

vectorstore = Chroma(embeddings=embeddings)
```
