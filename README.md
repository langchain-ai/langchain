# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

[![lint](https://github.com/hwchase17/langchain/actions/workflows/lint.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/lint.yml) [![test](https://github.com/hwchase17/langchain/actions/workflows/test.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



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

## 🚀 What can I do with this

This project was largely inspired by a few projects seen on Twitter for which we thought it would make sense to have more explicit tooling. A lot of the initial functionality was done in an attempt to recreate those. Those are:

**[Self-ask-with-search](https://ofir.io/self-ask.pdf)**

To recreate this paper, use the following code snippet or checkout the [example notebook](examples/self_ask_with_search.ipynb).

```
from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIChain

llm = OpenAI(temperature=0)
search = SerpAPIChain()

self_ask_with_search = SelfAskWithSearchChain(llm=llm, search_chain=search)

self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?")
```

**[LLM Math](https://twitter.com/amasad/status/1568824744367259648?s=20&t=-7wxpXBJinPgDuyHLouP1w)**
To recreate this example, use the following code snippet or check out the [example notebook](examples/llm_math.ipynb).

```
from langchain import OpenAI, LLMMathChain

llm = OpenAI(temperature=0)
llm_math = LLMMathChain(llm=llm)

llm_math.run("How many of the integers between 0 and 99 inclusive are divisible by 8?")
```