# langchain-ipfs

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-ipfs?label=%20)](https://pypi.org/project/langchain-ipfs/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-ipfs)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-ipfs)](https://pypistats.org/packages/langchain-ipfs)
[![Twitter](https://img.shields.io/twitter/url/https://twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-ipfs ipfs-pay-to-pin-client
```

## 🤔 What is this?

This package contains the LangChain integration with [ipfs-pay-to-pin](https://github.com/IcanBENCHurCAT/ipfs-pay-to-pin), a pay-to-pin IPFS storage service that enables decentralized pinning via x402 micropayments. It lets AI agents autonomously pin content to IPFS with support for multiple blockchains (Base L2, Solana, Algorand, Ethereum L1).

## 📖 Documentation

View the [documentation](https://docs.langchain.com/oss/python/integrations/providers/ipfs) for more details.

## Example

```python
from langchain_ipfs import IPFSPinTool

tool = IPFSPinTool()
result = tool.invoke({
    "content": "Hello, IPFS!",
    "pin_name": "my-first-pin",
    "expiration_days": 30
})
```

## Resources

- [LangChain Academy](https://academy.langchain.com/) — comprehensive, free courses on LangChain libraries and products
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
- [ipfs-pay-to-pin-client](https://github.com/IcanBENCHurCAT/ipfs-pay-to-pin) — SDK documentation
