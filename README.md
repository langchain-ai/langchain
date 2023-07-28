# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

[![Release Notes](https://img.shields.io/github/release/hwchase17/langchain)](https://github.com/hwchase17/langchain/releases)
[![CI](https://github.com/hwchase17/langchain/actions/workflows/langchain_ci.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/langchain_ci.yml)
[![Experimental CI](https://github.com/hwchase17/langchain/actions/workflows/langchain_experimental_ci.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/langchain_experimental_ci.yml)
[![Downloads](https://static.pepy.tech/badge/langchain/month)](https://pepy.tech/project/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/hwchase17/langchain)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hwchase17/langchain)
[![GitHub star chart](https://img.shields.io/github/stars/hwchase17/langchain?style=social)](https://star-history.com/#hwchase17/langchain)
[![Dependency Status](https://img.shields.io/librariesio/github/langchain-ai/langchain)](https://libraries.io/github/langchain-ai/langchain)
[![Open Issues](https://img.shields.io/github/issues-raw/hwchase17/langchain)](https://github.com/hwchase17/langchain/issues)


Looking for the JS/TS version? Check out [LangChain.js](https://github.com/hwchase17/langchainjs).

**Production Support:** As you move your LangChains into production, we'd love to offer more comprehensive support.
Please fill out [this form](https://6w1pwbss0py.typeform.com/to/rrbrdTH2) and we'll set up a dedicated support Slack channel.

## 🚨Breaking Changes for select chains (SQLDatabase) on 7/28

In an effort to make `langchain` leaner and safer, we are moving select chains to `langchain_experimental`.
This migration has already started, but we are remaining backwards compatible until 7/28.
On that date, we will remove functionality from `langchain`.
Read more about the motivation and the progress [here](https://github.com/hwchase17/langchain/discussions/8043).
Read how to migrate your code [here](MIGRATE.md).

## Quick Install

```bash
pip install langchain
# or
pip install langsmith && conda install langchain -c conda-forge
```
## Introduction

LangChain is a powerful library that enables developers to build applications using Large Language Models (LLMs). This repository serves as the core of LangChain and provides the tools and utilities to create composable LLM-based applications.

## Use Cases
LangChain can be used to build various applications. Here are some common examples:

## ❓ Question Answering over specific documents

- [Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- End-to-end Example: [Question Answering over Notion Database](https://github.com/hwchase17/notion-qa)

## 💬 Chatbots

- [Documentation](https://python.langchain.com/docs/use_cases/chatbots/)
- End-to-end Example: [Chat-LangChain](https://github.com/hwchase17/chat-langchain)

## 🤖 Agents
- [Documentation](https://python.langchain.com/docs/modules/agents/)
- End-to-end Example: [GPT+WolframAlpha](https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain)

## 📖 Documentation

Please refer to the [full documentation](https://python.langchain.com) for comprehensive details on using LangChain:

- Getting started (installation, setting up the environment, simple examples)
- How-To examples (demos, integrations, helper functions)
- API Reference: Detailed documentation of LangChain's API.
- Core Concepts: High-level explanation of key concepts.

## 🚀 What can this help with?

LangChain is designed to help with six main areas, each increasing in complexity:

**📃 LLMs and Prompts:**

Manage prompts, optimize prompts, provide a generic interface for all LLMs, and offer common utilities for working with LLMs.

**🔗 Chains:**

Go beyond a single LLM call and involve sequences of calls, whether to an LLM or a different utility. LangChain provides a standard interface for chains, integrates with other tools, and offers end-to-end chains for common applications.

**📚 Data Augmented Generation:**

Fetch data from external sources and use it in the generation step. Examples include summarization of long texts and question/answering over specific data sources.

**🤖 Agents:**

Let LLMs make decisions, take actions, observe results, and repeat until a task is complete. LangChain provides a standard interface for agents, a selection of agents, and examples of end-to-end agents.

**🧠 Memory:**

Persist state between calls of a chain/agent. LangChain provides a standard interface for memory, various memory implementations, and examples of chains/agents that use memory.

**🧐 Evaluation:**

[BETA] Generative models are notoriously hard to evaluate with traditional metrics.Evaluate generative models using language models themselves. LangChain provides prompts/chains to assist in this novel evaluation approach.

For more information on these concepts, please see our [full documentation](https://python.langchain.com).

## ⚙️ Configuration and Customization
LangChain can be configured to suit your specific needs and use cases. Explore the documentation to understand how to customize LangChain's behavior.

## 💁 Contributing

We warmly welcome contributions to the LangChain project. Whether it's new features, improved infrastructure, or better documentation, your efforts are appreciated.

For detailed information on how to contribute, see [CONTRIBUTING.md](.github/CONTRIBUTING.md).
