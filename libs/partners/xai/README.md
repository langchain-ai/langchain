# langchain-xai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-xai?label=%20)](https://pypi.org/project/langchain-xai/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-xai)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-xai)](https://pypistats.org/packages/langchain-xai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-xai
```

## 🤔 What is this?

This package contains the LangChain integrations for [xAI](https://x.ai/) through their [APIs](https://console.x.ai).

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_xai/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/xai).

## ⚠️ Important: API Migration

X.AI's Live Search API is being deprecated by **December 15, 2025**. Please migrate to the new agentic tool calling API:

```python
# Old (deprecated)
model = ChatXAI(search_parameters={"mode": "on"})

# New (recommended)
model = ChatXAI(server_tools=[{"type": "web_search"}])
```

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed migration instructions and [examples/agentic_tools_example.py](./examples/agentic_tools_example.py) for code examples.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
