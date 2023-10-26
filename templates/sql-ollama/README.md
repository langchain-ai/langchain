# SQL with LLaMA2 using llama.cpp

This template allows you to chat with a SQL database in natural language in private, using an open source LLM.

## Set up Ollama

Follow instructions [here](https://python.langchain.com/docs/integrations/chat/ollama) to download Ollama.

Also follow instructions to download your LLM of interest:

* This template uses `llama2:13b-chat`
* But you can pick from many LLMs [here](https://ollama.ai/library)

## Set up SQL DB

This template includes an example DB of 2023 NBA rosters.

You can see instructions to build this DB [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).

## Installation

```bash
# from inside your LangServe instance
poe add sql-ollama
```
