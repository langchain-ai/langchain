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
It aims to create:

1. a comprehensive collection of pieces you would ever want to combine
2. a flexible interface for combining pieces into a single comprehensive "chain"
3. a schema for easily saving and sharing those chains

## 📖 Documentation

Please see [here](https://langchain.readthedocs.io/en/latest/?) for full documentation on:
- Getting started (installation, setting up environment, simple examples)
- How-To examples (demos, integrations, helper functions)
- Reference (full API docs)
- Resources (high level explanation of core concepts)

## 🚀 What can I do with this

This project was largely inspired by a few projects seen on Twitter for which we thought it would make sense to have more explicit tooling. A lot of the initial functionality was done in an attempt to recreate those. Those are:

**[Self-ask-with-search](https://ofir.io/self-ask.pdf)**

To recreate this paper, use the following code snippet or checkout the [example notebook](https://github.com/hwchase17/langchain/blob/master/docs/examples/demos/self_ask_with_search.ipynb).

```python
from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIChain

llm = OpenAI(temperature=0)
search = SerpAPIChain()

self_ask_with_search = SelfAskWithSearchChain(llm=llm, search_chain=search)

self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?")
```

**[LLM Math](https://twitter.com/amasad/status/1568824744367259648?s=20&t=-7wxpXBJinPgDuyHLouP1w)**

To recreate this example, use the following code snippet or check out the [example notebook](https://github.com/hwchase17/langchain/blob/master/docs/examples/demos/llm_math.ipynb).

```python
from langchain import OpenAI, LLMMathChain

llm = OpenAI(temperature=0)
llm_math = LLMMathChain(llm=llm)

llm_math.run("How many of the integers between 0 and 99 inclusive are divisible by 8?")
```

**Generic Prompting**

You can also use this for simple prompting pipelines, as in the below example and this [example notebook](https://github.com/hwchase17/langchain/blob/master/docs/examples/demos/simple_prompts.ipynb).

```python
from langchain import Prompt, OpenAI, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = Prompt(template=template, input_variables=["question"])
llm = OpenAI(temperature=0)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.predict(question=question)
```

**Embed & Search Documents**

We support two vector databases to store and search embeddings -- FAISS and Elasticsearch. Here's a code snippet showing how to use FAISS to store embeddings and search for text similar to a query. Both database backends are featured in this [example notebook](https://github.com/hwchase17/langchain/blob/master/docs/examples/integrations/embeddings.ipynb).

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter

with open('state_of_the_union.txt') as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
```

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
