# langchain-box

This package contains the LangChain integration with Box

## Installation

```bash
pip install -U langchain-box
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatBox` class exposes chat models from Box.

```python
from langchain_box import ChatBox

llm = ChatBox()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`BoxEmbeddings` class exposes embeddings from Box.

```python
from langchain_box import BoxEmbeddings

embeddings = BoxEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`BoxLLM` class exposes LLMs from Box.

```python
from langchain_box import BoxLLM

llm = BoxLLM()
llm.invoke("The meaning of life is")
```
