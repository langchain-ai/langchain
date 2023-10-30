# RAG Weaviate

This template performs RAG using Weaviate and OpenAI.

##  Weaviate

This connects to a hosted Weaviate vectorstore.

Be sure that you have set a few env variables in `chain.py`:

* `WEAVIATE_ENVIRONMENT`
* `WEAVIATE_API_KEY`

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

## Environment variables

You need to define the following environment variables

```shell
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
WEAVIATE_ENVIRONMENT=<YOUR_WEAVIATE_ENVIRONMENT>
WEAVIATE_API_KEY=<YOUR_WEAVIATE_API_KEY>
```
