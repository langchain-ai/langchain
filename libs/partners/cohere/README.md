# LangChain Cohere

This package contains the Cohere integrations for LangChain.

## Installation

```bash
pip install langchain-cohere
```

## Usage

The `langchain-cohere` package provides integrations for Cohere's language models and embeddings.

### Chat Models

```python
from langchain_cohere import ChatCohere

chat = ChatCohere(model="command-r-plus")
```

### Embeddings

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

### Rerank

```python
from langchain_cohere import CohereRerank

rerank = CohereRerank(model="rerank-english-v3.0")
```

For more details, visit the [LangChain Cohere repository](https://github.com/langchain-ai/langchain-cohere).