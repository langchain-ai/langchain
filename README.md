# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

[![lint](https://github.com/hwchase17/langchain/actions/workflows/lint.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/lint.yml) [![test](https://github.com/hwchase17/langchain/actions/workflows/test.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai) [![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)

## Quick Install

`pip install langchain`

## 🤔 What is this?

Large language models (LLMs) are emerging as a transformative technology, enabling
developers to build applications that they previously could not.
But using these LLMs in isolation is often not enough to
create a truly powerful app - the real power comes when you are able to
combine them with other sources of computation or knowledge.

This library is aimed at assisting in the development of those types of applications.

## 📖 Documentation

Please see [here](https://langchain.readthedocs.io/en/latest/?) for full documentation on:
- Getting started (installation, setting up environment, simple examples)
- How-To examples (demos, integrations, helper functions)
- Reference (full API docs)
- Resources (high level explanation of core concepts)

## 🚀 What can this help with?

There are three main areas (with a forth coming soon) that LangChain is designed to help with.
These are, in increasing order of complexity:
1. LLM and Prompt usage
2. Chaining LLMs with other tools in a deterministic manner
3. Having a router LLM which uses other tools as needed
4. (Coming Soon) Memory

### LLMs and Prompts
Calling out to an LLM once is pretty easy, with most of them being behind well documented APIs.
However, there are still some challenges going from that to an application running in production that LangChain attempts to address:
- Easy switching costs: by exposing a standard interface for all the top LLM providers, LangChain makes it easy to switch from one provider to another, whether it be for production use cases or just for testing stuff out.
- Prompt management: managing your prompts is easy when you only have one simple one, but can get tricky when you have a bunch or when they start to get more complex. LangChain provides a standard way for storing, constructing, and referencing prompts.
- Prompt optimization: despite the underlying models getting better and better, there is still currently a need for carefully constructing prompts. 
- More coming soon

### Chains
Using an LLM in isolation is fine for some simple applications, but many more complex ones require chaining LLMs - either with eachother or with other tools.
LangChain provides several parts to help with that:
- Standard interface for working with Chains
- Easy way to construct chains of LLMs
- Lots of integrations with other tools that you may want to use in conjunction with LLMs (search, databases, Python REPL, etc)
- End-to-end chains for common workflows (database question/answer, recursive summarization, etc)

### Routing Chains
Some applications will require not just a predetermined chain of calls to LLMs/other tools, but potentially an unknown chain that depends on the user input.
In these types of chains, there is a "router" LLM chain which has access to a suite of tools.
Depending on the user input, the router can then decide which, if any, of these tools to call.
To help develop applications like these, LangChain provides:
- Standard router and router chain interfaces
- Common router LLM chains from literature
- Common chains that can be used as tools

### Memory
Coming soon.

## 🤖 Developer Guide

To begin developing on this project, first clone to the repo locally.
To install requirements, run `pip install -r requirements.txt`.
This will install all requirements for running the package, examples, linting, formatting, and tests.

Formatting for this project is a combination of [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/).
To run formatting for this project, run `make format`.

Linting for this project is a combination of [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.pycqa.org/en/latest/), and [mypy](http://mypy-lang.org/).
To run linting for this project, run `make lint`.
We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer and they can help you with it. We do not want this to be a blocker for good code getting contributed.

Unit tests cover modular logic that does not require calls to outside apis.
To run unit tests, run `make tests`.
If you add new logic, please add a unit test.

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).
To run integration tests, run `make integration_tests`.
If you add support for a new external API, please add a new integration test.

If you are adding a Jupyter notebook example, you can run `pip install -e .` to build the langchain package from your local changes, so your new logic can be imported into the notebook.

Docs are largely autogenerated by [sphinx](https://www.sphinx-doc.org/en/master/) from the code.
For that reason, we ask that you add good documentation to all classes and methods.
Similar to linting, we recognize documentation can be annoying - if you do not want to do it, please contact a project maintainer and they can help you with it. We do not want this to be a blocker for good code getting contributed.
