# langchain-ollama

This package contains the LangChain integration with Ollama

## Installation

```bash
pip install -U langchain-ollama
```

For the package to work, you will need to install and run the Ollama server locally ([download](https://ollama.com/download)).

To run integration tests (`make integration_tests`), you will need the following models installed in your Ollama server:

- `llama3`
- `llama3:latest`
- `lamma3.1`
- `gemma3:4b`
- `deepseek-r1:1.5b`

Install these models by running:

```bash
ollama pull <name-of-model>
```

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
