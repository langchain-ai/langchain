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

## Installation

Create your LangServe app:
```
langchain serve new my-app
cd my-app
```

Add template:
```
langchain serve add rag-pinecone
```

Start server:
```
langchain start
```

See Jupyter notebook `rag_pinecone` for various way to connect to the template.
