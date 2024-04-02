# langchain-cloudflare

This package contains the LangChain integration with Cloudflare

## Installation

```bash
pip install -U langchain-cloudflare
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatCloudflare` class exposes chat models from Cloudflare.

```python
from langchain_cloudflare import ChatCloudflare

llm = ChatCloudflare()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CloudflareEmbeddings` class exposes embeddings from Cloudflare.

```python
from langchain_cloudflare import CloudflareEmbeddings

embeddings = CloudflareEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`CloudflareLLM` class exposes LLMs from Cloudflare.

```python
from langchain_cloudflare import CloudflareLLM

llm = CloudflareLLM()
llm.invoke("The meaning of life is")
```
