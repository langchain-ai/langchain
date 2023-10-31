# RAG Astra

This template performs RAG using Astra and OpenAI.

## Astra

This connects to a vector-enabled Astra Database.

You should set the following environment variables from `chain.py`:

* `ASTRA_DB_APPLICATION_TOKEN`
* `ASTRA_DB_ID`
* `ASTRA_DB_COLLECTION_NAME`

## LLM

Be sure that `OPENAI_API_KEY` is set in order to access the OpenAI models.
