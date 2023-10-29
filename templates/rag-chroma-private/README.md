# Private RAG 

This template performs privae RAG (no reliance on external APIs) using:

* Ollama for the LLM
* GPT4All for embeddings

##  LLM

Follow instructions [here](https://python.langchain.com/docs/integrations/chat/ollama) to download Ollama.

The instructions also show how to download your LLM of interest with Ollama:

* This template uses `llama2:7b-chat`
* But you can pick from many [here](https://ollama.ai/library)

## Set up local embeddings

This will use [GPT4All](https://python.langchain.com/docs/integrations/text_embedding/gpt4all) embeddings.

##  Chroma

[Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) is an open-source vector database.

This template will create and add documents to the vector database in `chain.py`.

By default, this will load a popular blog post on agents.

However, you can choose from a large number of document loaders [here](https://python.langchain.com/docs/integrations/document_loaders).