# RAG Pinecone

This template performs RAG using Pinecone and OpenAI.

##  Pinecone

This connects to a hosted Pinecone vectorstore.

Be sure that you have set a few env variables in `chain.py`:

* `PINECONE_API_KEY`
* `PINECONE_ENV`
* `index_name`

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.
