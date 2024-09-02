# langchain-ollama

This package contains the LangChain integration with Ollama

## Installation

```bash
pip install -U langchain-ollama
```

You will also need to run the Ollama server locally. 
You can download it [here](https://ollama.com/download).

## Chat Models

`ChatOllama` class exposes chat models from Ollama.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3-groq-tool-use")
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OllamaEmbeddings` class exposes embeddings from Ollama.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`OllamaLLM` class exposes LLMs from Ollama.

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")
llm.invoke("The meaning of life is")
```
