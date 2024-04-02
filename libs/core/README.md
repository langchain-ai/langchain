# 🦜🍎️ LangChain Core

[![Downloads](https://static.pepy.tech/badge/langchain_core/month)](https://pepy.tech/project/langchain_core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Install

```bash
pip install langchain-core
```

## What is it?

LangChain Core contains the base abstractions that power the rest of the LangChain ecosystem.

These abstractions are designed to be as modular and simple as possible. Examples of these abstractions include those for language models, document loaders, embedding models, vectorstores, retrievers, and more.

The benefit of having these abstractions is that any provider can implement the required interface and then easily be used in the rest of the LangChain ecosystem.

For full documentation see the [API reference](https://api.python.langchain.com/en/stable/core_api_reference.html).

## 1️⃣ Core Interface: Runnables

The concept of a Runnable is central to LangChain Core – it is the interface that most LangChain Core components implement, giving them

- a common invocation interface (invoke, batch, stream, etc.)
- built-in utilities for retries, fallbacks, schemas and runtime configurability
- easy deployment with [LangServe](https://github.com/langchain-ai/langserve)

For more check out the [runnable docs](https://python.langchain.com/docs/expression_language/interface). Examples of components that implement the interface include: LLMs, Chat Models, Prompts, Retrievers, Tools, Output Parsers.

You can use LangChain Core objects in two ways:

1. **imperative**, ie. call them directly, eg. `model.invoke(...)`

2. **declarative**, with LangChain Expression Language (LCEL)

3. or a mix of both! eg. one of the steps in your LCEL sequence can be a custom function

| Feature   | Imperative                      | Declarative    |
| --------- | ------------------------------- | -------------- |
| Syntax    | All of Python                   | LCEL           |
| Tracing   | ✅ – Automatic                  | ✅ – Automatic |
| Parallel  | ✅ – with threads or coroutines | ✅ – Automatic |
| Streaming | ✅ – by yielding                | ✅ – Automatic |
| Async     | ✅ – by writing async functions | ✅ – Automatic |

## ⚡️ What is LangChain Expression Language?

LangChain Expression Language (LCEL) is a _declarative language_ for composing LangChain Core runnables into sequences (or DAGs), covering the most common patterns when building with LLMs.

LangChain Core compiles LCEL sequences to an _optimized execution plan_, with automatic parallelization, streaming, tracing, and async support.

For more check out the [LCEL docs](https://python.langchain.com/docs/expression_language/).

![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](../../docs/static/img/langchain_stack.png "LangChain Framework Overview")

For more advanced use cases, also check out [LangGraph](https://github.com/langchain-ai/langgraph), which is a graph-based runner for cyclic and recursive LLM workflows.

## 📕 Releases & Versioning

`langchain-core` is currently on version `0.1.x`.

As `langchain-core` contains the base abstractions and runtime for the whole LangChain ecosystem, we will communicate any breaking changes with advance notice and version bumps. The exception for this is anything in `langchain_core.beta`. The reason for `langchain_core.beta` is that given the rate of change of the field, being able to move quickly is still a priority, and this module is our attempt to do so.

Minor version increases will occur for:

- Breaking changes for any public interfaces NOT in `langchain_core.beta`

Patch version increases will occur for:

- Bug fixes
- New features
- Any changes to private interfaces
- Any changes to `langchain_core.beta`

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://python.langchain.com/docs/contributing/).

## ⛰️ Why build on top of LangChain Core?

The whole LangChain ecosystem is built on top of LangChain Core, so you're in good company when building on top of it. Some of the benefits:

- **Modularity**: LangChain Core is designed around abstractions that are independent of each other, and not tied to any specific model provider.
- **Stability**: We are committed to a stable versioning scheme, and will communicate any breaking changes with advance notice and version bumps.
- **Battle-tested**: LangChain Core components have the largest install base in the LLM ecosystem, and are used in production by many companies.
- **Community**: LangChain Core is developed in the open, and we welcome contributions from the community.
