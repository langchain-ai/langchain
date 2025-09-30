# 🦜🍎️ LangChain Core

[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-core)](https://pypistats.org/packages/langchain-core)

## Quick Install

```bash
pip install langchain-core
```

## What is it?

LangChain Core contains the base abstractions that power the the LangChain ecosystem.

These abstractions are designed to be as modular and simple as possible.

The benefit of having these abstractions is that any provider can implement the required interface and then easily be used in the rest of the LangChain ecosystem.

For full documentation see the [API reference](https://reference.langchain.com/python/).

## ⛰️ Why build on top of LangChain Core?

The LangChain ecosystem is built on top of `langchain-core`. Some of the benefits:

- **Modularity**: We've designed Core around abstractions that are independent of each other, and not tied to any specific model provider.
- **Stability**: We are committed to a stable versioning scheme, and will communicate any breaking changes with advance notice and version bumps.
- **Battle-tested**: Core components have the largest install base in the LLM ecosystem, and are used in production by many companies.

## 1️⃣ Core Interface: Runnables

The concept of a `Runnable` is central to LangChain Core – it is the interface that most LangChain Core components implement, giving them

- A common invocation interface (`invoke()`, `batch()`, `stream()`, etc.)
- Built-in utilities for retries, fallbacks, schemas and runtime configurability
- Easy deployment with [LangGraph](https://github.com/langchain-ai/langgraph)

For more check out the [`Runnable` docs](https://python.langchain.com/docs/concepts/runnables/). Examples of components that implement the interface include: Chat Models, Tools, Retrievers, and Output Parsers.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning Policy](https://docs.langchain.com/oss/python/versioning).

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing).
