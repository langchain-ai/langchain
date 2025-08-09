# langchain-superlinked

This package contains the LangChain integration with Superlinked

## Installation

```bash
pip install -U langchain-superlinked
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSuperlinked` class exposes chat models from Superlinked.

```python
from langchain_superlinked import ChatSuperlinked

llm = ChatSuperlinked()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SuperlinkedEmbeddings` class exposes embeddings from Superlinked.

```python
from langchain_superlinked import SuperlinkedEmbeddings

embeddings = SuperlinkedEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SuperlinkedLLM` class exposes LLMs from Superlinked.

```python
from langchain_superlinked import SuperlinkedLLM

llm = SuperlinkedLLM()
llm.invoke("The meaning of life is")
```
