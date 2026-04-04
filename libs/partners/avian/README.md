# langchain-avian

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-avian?label=%20)](https://pypi.org/project/langchain-avian/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-avian)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-avian)](https://pypistats.org/packages/langchain-avian)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-avian
```

## What is this?

This package contains the LangChain integration with [Avian](https://avian.io), an OpenAI-compatible inference API offering high-performance models including DeepSeek V3.2, Kimi K2.5, GLM-5, and MiniMax M2.5 (with 1M context support).

## Usage

```python
from langchain_avian import ChatAvian

llm = ChatAvian(
    model="deepseek-v3.2",
    temperature=0,
    # api_key="...",  # or set AVIAN_API_KEY env var
)
response = llm.invoke("Hello, how are you?")
```

### Available Models

| Model | Context Length | Input / Output (per M tokens) |
|---|---|---|
| `deepseek-v3.2` | 164K | $0.14 / $0.28 |
| `kimi-k2.5` | 128K | $0.14 / $0.28 |
| `glm-5` | 128K | $0.25 / $0.50 |
| `minimax-m2.5` | 1M | $0.15 / $0.30 |

## Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_avian/).

## Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
