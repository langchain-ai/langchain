# __package_name__

This package contains the LangChain integration with __ModuleName__

## Installation

```bash
pip install -U __package_name__
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`Chat__ModuleName__` class exposes chat models from __ModuleName__.

```python
from __module_name__ import Chat__ModuleName__

llm = Chat__ModuleName__()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`__ModuleName__Embeddings` class exposes embeddings from __ModuleName__.

```python
from __module_name__ import __ModuleName__Embeddings

embeddings = __ModuleName__Embeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`__ModuleName__LLM` class exposes LLMs from __ModuleName__.

```python
from __module_name__ import __ModuleName__LLM

llm = __ModuleName__LLM()
llm.invoke("The meaning of life is")
```
