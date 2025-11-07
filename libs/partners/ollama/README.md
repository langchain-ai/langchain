# langchain-ollama

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-ollama?label=%20)](https://pypi.org/project/langchain-ollama/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-ollama)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-ollama)](https://pypistats.org/packages/langchain-ollama)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-ollama
```

## 🤔 What is this?

This package contains the LangChain integration with Ollama

## 📖 Documentation

View the [documentation](https://docs.langchain.com/oss/python/integrations/providers/ollama) for more details.

## Development

### Running Tests

To run integration tests (`make integration_tests`), you will need the following models installed in your Ollama server:

- `llama3.1`
- `deepseek-r1:1.5b`
- `gpt-oss:20b`

Install these models by running:

```bash
ollama pull <name-of-model>
```
