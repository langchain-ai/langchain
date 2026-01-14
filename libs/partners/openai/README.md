# langchain-openai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-openai?label=%20)](https://pypi.org/project/langchain-openai/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-openai)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-openai)](https://pypistats.org/packages/langchain-openai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-openai
```

## 🤔 What is this?

This package contains the LangChain integrations for OpenAI through their `openai` SDK.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_openai/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/openai).

### High concurrency - optional OpenAI aiohttp backend

For improved throughput in high-concurrency scenarios (parallel chains, graphs, and agents), you can enable the OpenAI aiohttp backend which removes concurrency limits seen with the default httpx client.

**Installation:**
```bash
pip install "openai[aiohttp]"
```

**Usage:**
```python
from openai import DefaultAioHttpClient
from langchain_openai import ChatOpenAI

# Option 1: Pass explicitly
llm = ChatOpenAI(
    http_client=DefaultAioHttpClient(),
    http_async_client=DefaultAioHttpClient()
)

# Option 2: Use environment variable
# Set LC_OPENAI_USE_AIOHTTP=1 in your environment
llm = ChatOpenAI()  # Will automatically use aiohttp if available
```

For more details, see the [OpenAI Python library documentation](https://github.com/openai/openai-python#httpx-client).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
