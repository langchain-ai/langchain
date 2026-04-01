# langchain-shopsavvy

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-shopsavvy?label=%20)](https://pypi.org/project/langchain-shopsavvy/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-shopsavvy)](https://opensource.org/licenses/MIT)

## Quick Install

```bash
pip install langchain-shopsavvy
```

## What is this?

This package contains the LangChain integration with [ShopSavvy](https://shopsavvy.com), a price comparison platform with data on 100M+ products across thousands of retailers. It provides tools for product search, real-time price comparison, and price history analysis.

## Setup

Get a free API key at [shopsavvy.com/data](https://shopsavvy.com/data) and set it as an environment variable:

```bash
export SHOPSAVVY_API_KEY="ss_live_your_api_key"
```

## Tools

- **`ShopSavvyProductSearch`** — Search for products by keyword
- **`ShopSavvyPriceComparison`** — Get current prices from all retailers for a product
- **`ShopSavvyPriceHistory`** — Get historical price data to evaluate deals

## Retriever

- **`ShopSavvyRetriever`** — Retrieve product documents for use in RAG chains

## Documentation

- [ShopSavvy Data API docs](https://shopsavvy.com/data/documentation)
- [LangChain docs](https://docs.langchain.com)
