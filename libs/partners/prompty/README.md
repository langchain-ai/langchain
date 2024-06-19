# langchain-prompty

This package contains the LangChain integration with Microsoft Prompty.

## Installation

```bash
pip install -U langchain-prompty
```

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
