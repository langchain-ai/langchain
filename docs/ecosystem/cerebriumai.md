# CerebriumAI

This page covers how to use the CerebriumAI ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific CerebriumAI wrappers.

## Installation and Setup
- Install with `pip install cerebrium`
- Get an CerebriumAI api key and set it as an environment variable (`CEREBRIUMAI_API_KEY`)

## Wrappers

### LLM

There exists an CerebriumAI LLM wrapper, which you can access with 
```python
from langchain.llms import CerebriumAI
```