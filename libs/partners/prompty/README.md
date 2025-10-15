# langchain-prompty

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-prompty?label=%20)](https://pypi.org/project/langchain-prompty/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-prompty)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-prompty)](https://pypistats.org/packages/langchain-prompty)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-prompty
```

## 🤔 What is this?

This package contains the LangChain integration with Microsoft Prompty.

## 📖 Documentation

View the [documentation](https://docs.langchain.com/oss/python/integrations/providers/microsoft) for more details.

## Usage

Use the `create_chat_prompt` function to load `prompty` file as prompt.

```python
from langchain_prompty import create_chat_prompt

prompt = create_chat_prompt('<your .prompty file path>')
```

Then you can use the prompt for next steps.

Here is an example .prompty file:

```prompty
---
name: Basic Prompt
description: A basic prompt that uses the GPT-3 chat API to answer questions
authors:
  - author_1
  - author_2
model:
  api: chat
  configuration:
    azure_deployment: gpt-35-turbo
sample:
  firstName: Jane
  lastName: Doe
  question: What is the meaning of life?
  chat_history: []
---
system:
You are an AI assistant who helps people find information.
As the assistant, you answer questions briefly, succinctly,
and in a personable manner using markdown and even add some personal flair with appropriate emojis.

{% for item in chat_history %}
{{item.role}}:
{{item.content}}
{% endfor %}


user:
{{input}}

```
