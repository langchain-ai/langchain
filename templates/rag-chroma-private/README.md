# Private RAG 

This template create a private RAG server using:

* Ollama for the LLM
* GPT4All for embeddings

## Set up local LLM 

Follow instructions [here](https://python.langchain.com/docs/integrations/chat/ollama) to download Ollama.

Also follow instructions to download your LLM of interest:

* This template uses `llama2:13b-chat`
* But you can pick from many LLMs [here](https://ollama.ai/library)

## Set up local embeddings

This will use [GPT4All](https://python.langchain.com/docs/integrations/text_embedding/gpt4all) embeddings.

## Documents

This will load from a url of a popular blog post on agents.

However, you can choose from a large number of document loaders [here](https://python.langchain.com/docs/integrations/document_loaders).

## Installation

```bash
# from inside your LangServe instance
poe add chroma-rag-private
```