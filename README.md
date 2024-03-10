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

**LangChain** is a framework for developing applications powered by language models. It enables applications that:
- **Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- **Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

This framework consists of several parts.
- **LangChain Libraries**: The Python and JavaScript libraries. Contains interfaces and integrations for a myriad of components, a basic run time for combining these components into chains and agents, and off-the-shelf implementations of chains and agents.
- **[LangChain Templates](templates)**: A collection of easily deployable reference architectures for a wide variety of tasks.
- **[LangServe](https://github.com/langchain-ai/langserve)**: A library for deploying LangChain chains as a REST API.
- **[LangSmith](https://smith.langchain.com)**: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.
- **[LangGraph](https://python.langchain.com/docs/langgraph)**: LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. 

The LangChain libraries themselves are made up of several different packages.
- **[`langchain-core`](libs/core)**: Base abstractions and LangChain Expression Language.
- **[`langchain-community`](libs/community)**: Third party integrations.
- **[`langchain`](libs/langchain)**: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.

![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](docs/static/img/langchain_stack.png "LangChain Architecture Overview")

## 🧱 What can you build with LangChain?
**❓ Retrieval augmented generation**

- [Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- End-to-end Example: [Chat LangChain](https://chat.langchain.com) and [repo](https://github.com/langchain-ai/chat-langchain)

**💬 Analyzing structured data**

- [Documentation](https://python.langchain.com/docs/use_cases/qa_structured/sql)
- End-to-end Example: [SQL Llama2 Template](https://github.com/langchain-ai/langchain/tree/master/templates/sql-llama2)

**🤖 Chatbots**

- [Documentation](https://python.langchain.com/docs/use_cases/chatbots)
- End-to-end Example: [Web LangChain (web researcher chatbot)](https://weblangchain.vercel.app) and [repo](https://github.com/langchain-ai/weblangchain)

And much more! Head to the [Use cases](https://python.langchain.com/docs/use_cases/) section of the docs for more.

## 🚀 How does LangChain help?
The main value props of the LangChain libraries are:
1. **Components**: composable tools and integrations for working with language models. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not
2. **Off-the-shelf chains**: built-in assemblages of components for accomplishing higher-level tasks

Off-the-shelf chains make it easy to get started. Components make it easy to customize existing chains and build new ones. 

Components fall into the following **modules**:

**📃 Model I/O:**

This includes prompt management, prompt optimization, a generic interface for all LLMs, and common utilities for working with LLMs.

**📚 Retrieval:**

Data Augmented Generation involves specific types of chains that first interact with an external data source to fetch data for use in the generation step. Examples include summarization of long pieces of text and question/answering over specific data sources.

**🤖 Agents:**

Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end-to-end agents.

## 📖 Documentation

Please see [here](https://python.langchain.com) for full documentation, which includes:

- [Getting started](https://python.langchain.com/docs/get_started/introduction): installation, setting up the environment, simple examples
- Overview of the [interfaces](https://python.langchain.com/docs/expression_language/), [modules](https://python.langchain.com/docs/modules/), and [integrations](https://python.langchain.com/docs/integrations/providers)
- [Use case](https://python.langchain.com/docs/use_cases/qa_structured/sql) walkthroughs and best practice [guides](https://python.langchain.com/docs/guides/adapters/openai)
- [LangSmith](https://python.langchain.com/docs/langsmith/), [LangServe](https://python.langchain.com/docs/langserve), and [LangChain Template](https://python.langchain.com/docs/templates/) overviews
- [Reference](https://api.python.langchain.com): full API docs


## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see [here](https://python.langchain.com/docs/contributing/).

## 🌟 Contributors

[![langchain contributors](https://contrib.rocks/image?repo=langchain-ai/langchain&max=2000)](https://github.com/langchain-ai/langchain/graphs/contributors)
