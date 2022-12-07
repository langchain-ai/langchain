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

There are four main areas that LangChain is designed to help with.
These are, in increasing order of complexity:
1. LLM and Prompts
2. Chains
3. Agents
4. Memory

Let's go through these categories and for each one identify key concepts (to clarify terminology) as well as the problems in this area LangChain helps solve.

### LLMs and Prompts
Calling out to an LLM once is pretty easy, with most of them being behind well documented APIs.
However, there are still some challenges going from that to an application running in production that LangChain attempts to address.

**Key Concepts**
- LLM: A large language model, in particular a text-to-text model.
- Prompt: The input to a language model. Typically this is not simply a hardcoded string but rather a combination of a template, some examples, and user input.
- Prompt Template: An object responsible for constructing the final prompt to pass to a LLM.
- Examples: Datapoints that can be included in the prompt in order to give the model more context what to do.
- Few Shot Prompt Template: A subclass of the PromptTemplate class that uses examples.
- Example Selector: A class responsible to selecting examples to use dynamically (depending on user input) in a few shot prompt.

**Problems Solved**
- Switching costs: by exposing a standard interface for all the top LLM providers, LangChain makes it easy to switch from one provider to another, whether it be for production use cases or just for testing stuff out.
- Prompt management: managing your prompts is easy when you only have one simple one, but can get tricky when you have a bunch or when they start to get more complex. LangChain provides a standard way for storing, constructing, and referencing prompts.
- Prompt optimization: despite the underlying models getting better and better, there is still currently a need for carefully constructing prompts. 

### Chains
Using an LLM in isolation is fine for some simple applications, but many more complex ones require chaining LLMs - either with eachother or with other experts.
LangChain provides several parts to help with that.

**Key Concepts**
- Tools: APIs designed for assisting with a particular use case (search, databases, Python REPL, etc). Prompt templates, LLMs, and chains can also be considered tools.
- Chains: A combination of multiple tools in a deterministic manner.

**Problems Solved**
- Standard interface for working with Chains
- Easy way to construct chains of LLMs
- Lots of integrations with other tools that you may want to use in conjunction with LLMs 
- End-to-end chains for common workflows (database question/answer, recursive summarization, etc)

### Agents
Some applications will require not just a predetermined chain of calls to LLMs/other tools, but potentially an unknown chain that depends on the user input.
In these types of chains, there is a “agent” which has access to a suite of tools.
Depending on the user input, the agent can then decide which, if any, of these tools to call.

**Key Concepts**
- Tools: same as above.
- Agent: An LLM-powered class responsible for determining which tools to use and in what order.


**Problems Solved**
- Standard agent interfaces
- A selection of powerful agents to choose from
- Common chains that can be used as tools

### Memory
By default, Chains and Agents are stateless, meaning that they treat each incoming query independently.
In some applications (chatbots being a GREAT example) it is highly important to remember previous interactions,
both at a short term but also at a long term level. The concept of "Memory" exists to do exactly that.

**Key Concepts**
- Memory: A class that can be added to an Agent or Chain to (1) pull in memory variables before calling that chain/agent, and (2) create new memories after the chain/agent finishes.
- Memory Variables: Variables returned from a Memory class, to be passed into the chain/agent along with the user input.

**Problems Solved**
- Standard memory interfaces
- A collection of common memory implementations to choose from
- Common chains/agents that use memory (e.g. chatbots)

## 🤖 Developer Guide

To begin developing on this project, first clone the repo locally.

### Quick Start

This project uses [Poetry](https://python-poetry.org/) as a dependency manager. Check out Poetry's own [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```bash
poetry install -E all
```

This will install all requirements for running the package, examples, linting, formatting, and tests. Note the `-E all` flag will install all optional dependencies necessary for integration testing.

Now, you should be able to run the common tasks in the following section.

### Common Tasks

#### Code Formatting

Formatting for this project is a combination of [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/).

To run formatting for this project:

```bash
make format
```

#### Linting

Linting for this project is a combination of [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.pycqa.org/en/latest/), and [mypy](http://mypy-lang.org/).

To run linting for this project:

```bash
make lint
```

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer and they can help you with it. We do not want this to be a blocker for good code getting contributed.

#### Testing

Unit tests cover modular logic that does not require calls to outside apis.

To run unit tests:

```bash
make tests
```

If you add new logic, please add a unit test.

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).

To run integration tests:

```bash
make integration_tests
```

If you add support for a new external API, please add a new integration test.

#### Adding a Jupyter Notebook

If you are adding a Jupyter notebook example, you'll want to install the optional `dev` dependencies.

To install dev dependencies:

```bash
poetry install --with dev
```

Launch a notebook:

```bash
poetry run jupyter notebook
```

When you run `poetry install`, the `langchain` package is installed as editable in the virtualenv, so your new logic can be imported into the notebook.

#### Contribute Documentation

Docs are largely autogenerated by [sphinx](https://www.sphinx-doc.org/en/master/) from the code.

For that reason, we ask that you add good documentation to all classes and methods.

Similar to linting, we recognize documentation can be annoying - if you do not want to do it, please contact a project maintainer and they can help you with it. We do not want this to be a blocker for good code getting contributed.
