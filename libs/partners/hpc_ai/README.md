# langchain-hpc-ai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-hpc-ai?label=%20)](https://pypi.org/project/langchain-hpc-ai/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-hpc-ai)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-hpc-ai)](https://pypistats.org/packages/langchain-hpc-ai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Use [`ChatOpenAI`](https://js.langchain.com/docs/integrations/chat/openai) from `@langchain/openai` with a custom base URL, or check [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-hpc-ai
```

## 🤔 What is this?

This package contains the LangChain integration with [HPC-AI](https://www.hpc-ai.com/) inference (OpenAI-compatible API).

## Environment variables

- `HPC_AI_API_KEY` — API key (required when using the default base URL).
- `HPC_AI_BASE_URL` — Optional override; defaults to `https://api.hpc-ai.com/inference/v1`.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_hpc_ai/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/hpc-ai).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
