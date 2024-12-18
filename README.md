# 🦜️🔗 LangChain

⚡ Build context-aware reasoning applications ⚡

[![Release Notes](https://img.shields.io/github/release/langchain-ai/langchain?style=flat-square)](https://github.com/langchain-ai/langchain/releases)
[![CI](https://github.com/langchain-ai/langchain/actions/workflows/check_diffs.yml/badge.svg)](https://github.com/langchain-ai/langchain/actions/workflows/check_diffs.yml)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-core?style=flat-square)](https://pypistats.org/packages/langchain-core)
[![GitHub star chart](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat-square)](https://star-history.com/#langchain-ai/langchain)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain?style=flat-square)](https://github.com/langchain-ai/langchain/issues)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode&style=flat-square)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/langchain-ai/langchain)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/langchain)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS library? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

To help you ship LangChain apps to production faster, check out [LangSmith](https://smith.langchain.com).
[LangSmith](https://smith.langchain.com) is a unified developer platform for building, testing, and monitoring LLM applications.
Fill out [this form](https://www.langchain.com/contact-sales) to speak with our sales team.

## Quick Install

With pip:

```bash
pip install langchain
```

With conda:

```bash
conda install langchain -c conda-forge
```

## 🤔 What is LangChain?

**LangChain** is a framework for developing applications powered by large language models (LLMs).

For these applications, LangChain simplifies the entire application lifecycle:


- **Open-source libraries**: Build your applications using LangChain's open-source
[components](https://python.langchain.com/docs/concepts/) and
[third-party integrations](https://python.langchain.com/docs/integrations/providers/).
  Use [LangGraph](https://langchain-ai.github.io/langgraph/) to build stateful agents with first-class streaming and human-in-the-loop support.
- **Productionization**: Inspect, monitor, and evaluate your apps with [LangSmith](https://docs.smith.langchain.com/) so that you can constantly optimize and deploy with confidence.
- **Deployment**: Turn your LangGraph applications into production-ready APIs and Assistants with [LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/).

### Open-source libraries

- **`langchain-core`**: Base abstractions.
- **Integration packages** (e.g. **`langchain-openai`**, **`langchain-anthropic`**, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.
- **`langchain`**: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
- **`langchain-community`**: Third-party integrations that are community maintained.
- **[LangGraph](https://langchain-ai.github.io/langgraph)**: Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph. Integrates smoothly with LangChain, but can be used without it. To learn more about LangGraph, check out our first LangChain Academy course, *Introduction to LangGraph*, available [here](https://academy.langchain.com/courses/intro-to-langgraph).

### Productionization:

- **[LangSmith](https://docs.smith.langchain.com/)**: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.

### Deployment:

- **[LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/)**: Turn your LangGraph applications into production-ready APIs and Assistants.

![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](docs/static/svg/langchain_stack_112024.svg#gh-light-mode-only "LangChain Architecture Overview")
![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](docs/static/svg/langchain_stack_112024_dark.svg#gh-dark-mode-only "LangChain Architecture Overview")

## 🧱 What can you build with LangChain?

**❓ Question answering with RAG**

- [Documentation](https://python.langchain.com/docs/tutorials/rag/)
- End-to-end Example: [Chat LangChain](https://chat.langchain.com) and [repo](https://github.com/langchain-ai/chat-langchain)

**🧱 Extracting structured output**

- [Documentation](https://python.langchain.com/docs/tutorials/extraction/)
- End-to-end Example: [LangChain Extract](https://github.com/langchain-ai/langchain-extract/)

**🤖 Chatbots**

- [Documentation](https://python.langchain.com/docs/tutorials/chatbot/)
- End-to-end Example: [Web LangChain (web researcher chatbot)](https://weblangchain.vercel.app) and [repo](https://github.com/langchain-ai/weblangchain)

And much more! Head to the [Tutorials](https://python.langchain.com/docs/tutorials/) section of the docs for more.

## 🚀 How does LangChain help?

The main value props of the LangChain libraries are:

1. **Components**: composable building blocks, tools and integrations for working with language models. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not.
2. **Easy orchestration with LangGraph**: [LangGraph](https://langchain-ai.github.io/langgraph/),
built on top of `langchain-core`, has built-in support for [messages](https://python.langchain.com/docs/concepts/messages/), [tools](https://python.langchain.com/docs/concepts/tools/),
and other LangChain abstractions. This makes it easy to combine components into
production-ready applications with persistence, streaming, and other key features.
Check out the LangChain [tutorials page](https://python.langchain.com/docs/tutorials/#orchestration) for examples.

## Components

Components fall into the following **modules**:

**📃 Model I/O**

This includes [prompt management](https://python.langchain.com/docs/concepts/prompt_templates/)
and a generic interface for [chat models](https://python.langchain.com/docs/concepts/chat_models/), including a consistent interface for [tool-calling](https://python.langchain.com/docs/concepts/tool_calling/) and [structured output](https://python.langchain.com/docs/concepts/structured_outputs/) across model providers.

**📚 Retrieval**

Retrieval Augmented Generation involves [loading data](https://python.langchain.com/docs/concepts/document_loaders/) from a variety of sources, [preparing it](https://python.langchain.com/docs/concepts/text_splitters/), then [searching over (a.k.a. retrieving from)](https://python.langchain.com/docs/concepts/retrievers/) it for use in the generation step.

**🤖 Agents**

Agents allow an LLM autonomy over how a task is accomplished. Agents make decisions about which Actions to take, then take that Action, observe the result, and repeat until the task is complete. [LangGraph](https://langchain-ai.github.io/langgraph/) makes it easy to use
LangChain components to build both [custom](https://langchain-ai.github.io/langgraph/tutorials/)
and [built-in](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
LLM agents.

## 📖 Documentation

Please see [here](https://python.langchain.com) for full documentation, which includes:

- [Introduction](https://python.langchain.com/docs/introduction/): Overview of the framework and the structure of the docs.
- [Tutorials](https://python.langchain.com/docs/tutorials/): If you're looking to build something specific or are more of a hands-on learner, check out our tutorials. This is the best place to get started.
- [How-to guides](https://python.langchain.com/docs/how_to/): Answers to “How do I….?” type questions. These guides are goal-oriented and concrete; they're meant to help you complete a specific task.
- [Conceptual guide](https://python.langchain.com/docs/concepts/): Conceptual explanations of the key parts of the framework.
- [API Reference](https://python.langchain.com/api_reference/): Thorough documentation of every class and method.

## 🌐 Ecosystem

- [🦜🛠️ LangSmith](https://docs.smith.langchain.com/): Trace and evaluate your language model applications and intelligent agents to help you move from prototype to production.
- [🦜🕸️ LangGraph](https://langchain-ai.github.io/langgraph/): Create stateful, multi-actor applications with LLMs. Integrates smoothly with LangChain, but can be used without it.
- [🦜🕸️ LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#langgraph-platform): Deploy LLM applications built with LangGraph into production.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see [here](https://python.langchain.com/docs/contributing/).

## 🌟 Contributors

[![langchain contributors](https://contrib.rocks/image?repo=langchain-ai/langchain&max=2000)](https://github.com/langchain-ai/langchain/graphs/contributors)
