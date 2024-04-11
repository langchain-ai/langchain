# 🦜️🔗 LangChain

⚡ Build context-aware reasoning applications ⚡

[![Release Notes](https://img.shields.io/github/release/langchain-ai/langchain)](https://github.com/langchain-ai/langchain/releases)
[![CI](https://github.com/langchain-ai/langchain/actions/workflows/check_diffs.yml/badge.svg)](https://github.com/langchain-ai/langchain/actions/workflows/check_diffs.yml)
[![Downloads](https://static.pepy.tech/badge/langchain/month)](https://pepy.tech/project/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/langchain-ai/langchain)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/langchain)
[![GitHub star chart](https://img.shields.io/github/stars/langchain-ai/langchain?style=social)](https://star-history.com/#langchain-ai/langchain)
[![Dependency Status](https://img.shields.io/librariesio/github/langchain-ai/langchain)](https://libraries.io/github/langchain-ai/langchain)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain)](https://github.com/langchain-ai/langchain/issues)

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

- **Open-source libraries**: Build your applications using LangChain's [modular building blocks](https://python.langchain.com/docs/expression_language/) and [components](https://python.langchain.com/docs/modules/). Integrate with hundreds of [third-party providers](https://python.langchain.com/docs/integrations/platforms/).
- **Productionization**: Inspect, monitor, and evaluate your apps with [LangSmith](https://python.langchain.com/docs/langsmith/) so that you can constantly optimize and deploy with confidence.
- **Deployment**: Turn any chain into a REST API with [LangServe](https://python.langchain.com/docs/langserve).

### Open-source libraries
- **`langchain-core`**: Base abstractions and LangChain Expression Language.
- **`langchain-community`**: Third party integrations.
  - Some integrations have been further split into **partner packages** that only rely on **`langchain-core`**. Examples include **`langchain_openai`** and **`langchain_anthropic`**.
- **`langchain`**: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
- **`[LangGraph](https://python.langchain.com/docs/langgraph)`**: A library for building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

### Productionization:
- **[LangSmith](https://python.langchain.com/docs/langsmith)**: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.

### Deployment:
- **[LangServe](https://python.langchain.com/docs/langserve)**: A library for deploying LangChain chains as REST APIs.

![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](docs/static/svg/langchain_stack.svg "LangChain Architecture Overview")

## 🧱 What can you build with LangChain?

**❓ Question answering with RAG**

- [Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- End-to-end Example: [Chat LangChain](https://chat.langchain.com) and [repo](https://github.com/langchain-ai/chat-langchain)

**🧱 Extracting structured output**

- [Documentation](https://python.langchain.com/docs/use_cases/extraction/)
- End-to-end Example: [SQL Llama2 Template](https://github.com/langchain-ai/langchain-extract/)

**🤖 Chatbots**

- [Documentation](https://python.langchain.com/docs/use_cases/chatbots)
- End-to-end Example: [Web LangChain (web researcher chatbot)](https://weblangchain.vercel.app) and [repo](https://github.com/langchain-ai/weblangchain)

And much more! Head to the [Use cases](https://python.langchain.com/docs/use_cases/) section of the docs for more.

## 🚀 How does LangChain help?
The main value props of the LangChain libraries are:
1. **Components**: composable building blocks, tools and integrations for working with language models. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not
2. **Off-the-shelf chains**: built-in assemblages of components for accomplishing higher-level tasks

Off-the-shelf chains make it easy to get started. Components make it easy to customize existing chains and build new ones. 

## LangChain Expression Language (LCEL)

LCEL is the foundation of many of LangChain's components, and is a declarative way to compose chains. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains.

- **[Overview](https://python.langchain.com/docs/expression_language/)**: LCEL and its benefits
- **[Interface](https://python.langchain.com/docs/expression_language/interface)**: The standard interface for LCEL objects
- **[Primitives](https://python.langchain.com/docs/expression_language/primitives)**: More on the primitives LCEL includes

## Components

Components fall into the following **modules**:

**📃 Model I/O:**

This includes [prompt management](https://python.langchain.com/docs/modules/model_io/prompts/), [prompt optimization](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/), a generic interface for [chat models](https://python.langchain.com/docs/modules/model_io/chat/) and [LLMs](https://python.langchain.com/docs/modules/model_io/llms/), and common utilities for working with [model outputs](https://python.langchain.com/docs/modules/model_io/output_parsers/).

**📚 Retrieval:**

Retrieval Augmented Generation involves [loading data](https://python.langchain.com/docs/modules/data_connection/document_loaders/) from a variety of sources, [preparing it](https://python.langchain.com/docs/modules/data_connection/document_loaders/), [then retrieving it](https://python.langchain.com/docs/modules/data_connection/retrievers/) for use in the generation step.

**🤖 Agents:**

Agents allow an LLM autonomy over how a task is accomplished. Agents make decisions about which Actions to take, then take that Action, observe the result, and repeat until the task is complete done. LangChain provides a [standard interface for agents](https://python.langchain.com/docs/modules/agents/), a [selection of agents](https://python.langchain.com/docs/modules/agents/agent_types/) to choose from, and examples of end-to-end agents.

## 📖 Documentation

Please see [here](https://python.langchain.com) for full documentation, which includes:

- [Getting started](https://python.langchain.com/docs/get_started/introduction): installation, setting up the environment, simple examples
- [Use case](https://python.langchain.com/docs/use_cases/) walkthroughs and best practice [guides](https://python.langchain.com/docs/guides/)
- Overviews of the [interfaces](https://python.langchain.com/docs/expression_language/), [components](https://python.langchain.com/docs/modules/), and [integrations](https://python.langchain.com/docs/integrations/providers)

You can also check out the full [API Reference docs](https://api.python.langchain.com).

## 🌐 Ecosystem

- [🦜🛠️ LangSmith](https://python.langchain.com/docs/langsmith/): Tracing and evaluating your language model applications and intelligent agents to help you move from prototype to production.
- [🦜🕸️ LangGraph](https://python.langchain.com/docs/langgraph): Creating stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain primitives.
- [🦜🏓 LangServe](https://python.langchain.com/docs/langserve): Deploying LangChain runnables and chains as REST APIs.
  - [LangChain Templates](https://python.langchain.com/docs/templates/): Example applications hosted with LangServe.


## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see [here](https://python.langchain.com/docs/contributing/).

## 🌟 Contributors

[![langchain contributors](https://contrib.rocks/image?repo=langchain-ai/langchain&max=2000)](https://github.com/langchain-ai/langchain/graphs/contributors)
