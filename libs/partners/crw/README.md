# langchain-crw

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-crw?label=%20)](https://pypi.org/project/langchain-crw/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-crw)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-crw)](https://pypistats.org/packages/langchain-crw)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-crw
```

## What is this?

This package contains the LangChain integration with [CRW](https://github.com/crw-rs/crw), a high-performance, Firecrawl-compatible web scraper written in Rust.

CRW can be self-hosted via `crw-server` or used as a cloud service at [fastcrw.com](https://fastcrw.com).

## Usage

```python
from langchain_crw import CrwLoader

# Self-hosted (no API key needed)
loader = CrwLoader(url="https://example.com", mode="scrape")

# Cloud
loader = CrwLoader(
    url="https://example.com",
    api_key="fc-...",
    api_url="https://fastcrw.com/api",
    mode="crawl",
)

docs = loader.load()
```
