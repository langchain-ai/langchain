# langchain-naver

This package contains the LangChain integration with Naver

## Installation

```bash
pip install -U langchain-naver
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatNaver` class exposes chat models from Naver.

```python
from langchain_naver import ChatNaver

llm = ChatNaver()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`NaverEmbeddings` class exposes embeddings from Naver.

```python
from langchain_naver import NaverEmbeddings

embeddings = NaverEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```
