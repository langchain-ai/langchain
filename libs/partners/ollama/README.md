# langchain-ollama

This package contains the LangChain integration with Ollama

## Installation

```bash
pip install -U langchain-ollama
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatOllama` class exposes chat models from Ollama.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OllamaEmbeddings` class exposes embeddings from Ollama.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`OllamaLLM` class exposes LLMs from Ollama.

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM()
llm.invoke("The meaning of life is")
```
