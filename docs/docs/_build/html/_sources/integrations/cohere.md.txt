# Cohere

>[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models
> that help companies improve human-machine interactions.

## Installation and Setup
- Install the Python SDK :
```bash
pip install cohere
```

Get a [Cohere api key](https://dashboard.cohere.ai/) and set it as an environment variable (`COHERE_API_KEY`)


## LLM

There exists an Cohere LLM wrapper, which you can access with 
See a [usage example](../modules/models/llms/integrations/cohere.ipynb).

```python
from langchain.llms import Cohere
```

## Text Embedding Model

There exists an Cohere Embedding model, which you can access with 
```python
from langchain.embeddings import CohereEmbeddings
```
For a more detailed walkthrough of this, see [this notebook](../modules/models/text_embedding/examples/cohere.ipynb)

## Retriever

See a [usage example](../modules/indexes/retrievers/examples/cohere-reranker.ipynb).

```python
from langchain.retrievers.document_compressors import CohereRerank
```
