# Shale Protocol

[Shale Protocol](https://shaleprotocol.com) provides free and production-ready LLMs APIs to accelerate researches and innovations based on open LLMs.

With Shale Protocol, developers/resaerchers can create apps and explore the power of open LLMs with no cost.

This page covers how use Shale Protocol as an LLM back-end with LangChain.


## How to

### 1. Get an API key at https://shaleprotocol.com for free

### 2. Use https://shale.live/v1 as OpenAI API drop-in replacement 

For example
```python
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

import os
os.environ['OPENAI_API_BASE'] = "https://shale.live/v1"
os.environ['OPENAI_API_KEY'] = "ENTER YOUR API KEY"

llm = OpenAI()

template = """Question: {question}

# Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

```