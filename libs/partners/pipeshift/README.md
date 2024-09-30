# langchain-pipeshift

This package contains the LangChain integration with [Pipeshift](https://pipeshift.com)

## Installation
- Start by installing the package:
```bash
pip install -U langchain-pipeshift
```

## Environment Variables
- Head on to [Pipeshift Dashboard](https://dashboard.pipeshift.com) to get your API key.
- Add the api key to your environment variables:
```bash
PIPESHIFT_API_KEY=<your-api-key>
```

## Chat Models

`ChatPipeshift` class exposes chat models from Pipeshift.

```python
from langchain_pipeshift import ChatPipeshift

llm = ChatPipeshift()
llm.invoke("Sing a ballad of LangChain.")
```

<!-- ## Embeddings

`PipeshiftEmbeddings` class exposes embeddings from Pipeshift.

```python
from langchain_pipeshift import PipeshiftEmbeddings

embeddings = PipeshiftEmbeddings()
embeddings.embed_query("What is the meaning of life?")
``` -->

## LLMs
`Pipeshift` class exposes LLMs from Pipeshift.

```python
from langchain_pipeshift import Pipeshift

llm = Pipeshift()
llm.invoke("The meaning of life is")
```
