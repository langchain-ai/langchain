# langchain-unstructured

This package contains the LangChain integration with Unstructured

## Installation

```bash
pip install -U langchain-unstructured
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatUnstructured` class exposes chat models from Unstructured.

```python
from langchain_unstructured import ChatUnstructured

llm = ChatUnstructured()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`UnstructuredEmbeddings` class exposes embeddings from Unstructured.

```python
from langchain_unstructured import UnstructuredEmbeddings

embeddings = UnstructuredEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`UnstructuredLLM` class exposes LLMs from Unstructured.

```python
from langchain_unstructured import UnstructuredLLM

llm = UnstructuredLLM()
llm.invoke("The meaning of life is")
```
