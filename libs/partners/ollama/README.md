# langchain-ollama

This package contains the LangChain integration with Ollama

## Installation

```bash
pip install -U langchain-ollama
```

For the package to work, you will need to install and run the Ollama server locally ([download](https://ollama.com/download)).

To run integration tests (`make integration_tests`), you will need the following models installed in your Ollama server:

- `llama3.1`
- `deepseek-r1:1.5b`

Install these models by running:

```bash
ollama pull <name-of-model>
```

## [Chat Models](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#chatollama)

`ChatOllama` class exposes chat models from Ollama.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1")
llm.invoke("Sing a ballad of LangChain.")
```

## [Embeddings](https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html#ollamaembeddings)

`OllamaEmbeddings` class exposes embeddings from Ollama.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3.1")
embeddings.embed_query("What is the meaning of life?")
```

## [LLMs](https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html#ollamallm)

`OllamaLLM` class exposes traditional LLMs from Ollama.

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")
llm.invoke("The meaning of life is")
```
