# langchain-minimax

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-minimax?label=%20)](https://pypi.org/project/langchain-minimax/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-minimax)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-minimax
```

## What is this?

This package contains the LangChain integration with [MiniMax](https://www.minimaxi.com/).

MiniMax provides large language models accessible through an OpenAI-compatible API,
including MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.5, and MiniMax-M2.5-highspeed.

## Usage

```python
from langchain_minimax import ChatMiniMax

model = ChatMiniMax(
    model="MiniMax-M2.7",
    # api_key="...",  # or set MINIMAX_API_KEY env var
)
response = model.invoke("Hello, how are you?")
print(response.content)
```

## Documentation

For full documentation, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/).

## Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
