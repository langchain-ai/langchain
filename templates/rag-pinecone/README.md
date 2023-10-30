# RAG Pinecone

This template performs RAG using Pinecone and OpenAI.

##  Pinecone

This connects to a hosted Pinecone vectorstore.

Be sure that you have set a few env variables in `chain.py`:

* `PINECONE_API_KEY`
* `PINECONE_ENV`
* `PINECONE_INDEX`

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

## Environment variables

You need to define the following environment variables

```shell
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
PINECONE_API_KEY=<YOUR_PINECONE_API_KEY>
PINECONE_ENVIRONMENT=<YOUR_PINECONE_INDEX>
PINECONE_INDEX=<YOUR_PINECONE_INDEX>
```
