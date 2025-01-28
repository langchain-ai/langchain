# langchain-deepseek

This package contains the LangChain integration with DeepSeek

## Installation

```bash
pip install -U langchain-deepseek
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatDeepSeek` class exposes chat models from DeepSeek.

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`DeepSeekEmbeddings` class exposes embeddings from DeepSeek.

```python
from langchain_deepseek import DeepSeekEmbeddings

embeddings = DeepSeekEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`DeepSeekLLM` class exposes LLMs from DeepSeek.

```python
from langchain_deepseek import DeepSeekLLM

llm = DeepSeekLLM()
llm.invoke("The meaning of life is")
```
