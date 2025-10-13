# 🦜✂️ LangChain Text Splitters

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-text-splitters?label=%20)](https://pypi.org/project/langchain-text-splitters/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-text-splitters)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-text-splitters)](https://pypistats.org/packages/langchain-text-splitters)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-text-splitters
```

## 🤔 What is this?

LangChain Text Splitters contains utilities for splitting into chunks a wide variety of text documents.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/langchain).

## 🛠️ Troubleshooting: `chunk_overlap` seems not to apply

- After header based splitting (e.g., `MarkdownHeaderTextSplitter`), use **`split_documents(docs)`** (not `split_text`) so overlap is applied **within each section** and per section metadata (headers) is preserved on chunks.
- Overlap appears only when a **single input section** exceeds `chunk_size` and is split into multiple chunks.
- Overlap **does not cross** section/document boundaries (e.g., `# H1` → `## H2`).
- If the header becomes a tiny first chunk, there's nothing meaningful to overlap. Consider `strip_headers=True` in `MarkdownHeaderTextSplitter`, or reduce separators so the section forms a longer segment.
- If your text lacks newlines/spaces, keep a fallback `""` in `separators` so the splitter can still split and apply overlap.

> Looking for examples and API details? See the [Text Splitters how-to](https://python.langchain.com/docs/how_to/#text-splitters) and the [API reference](https://python.langchain.com/api_reference/text_splitters/index.html).


## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

We encourage pinning your version to a specific version in order to avoid breaking your CI when we publish new tests. We recommend upgrading to the latest version periodically to make sure you have the latest tests.

Not pinning your version will ensure you always have the latest tests, but it may also break your CI if we introduce tests that your integration doesn't pass.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
