# langchain-perplexity

This package contains the LangChain integration with Perplexity.

## Installation

```bash
pip install -U langchain-perplexity
```

And you should [configure your perplexity credentials](https://docs.perplexity.ai/guides/getting-started)
and then set the `PPLX_API_KEY` environment variable.

## Usage

This package contains the `ChatPerplexity` class, which is the recommended way to interface with Perplexity chat models.

```python
import getpass
import os

if not os.environ.get("PPLX_API_KEY"):
  os.environ["PPLX_API_KEY"] = getpass.getpass("Enter API key for Perplexity: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("llama-3.1-sonar-small-128k-online", model_provider="perplexity")
llm.invoke("Hello, world!")
```
