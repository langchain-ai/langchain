# RAG Chroma

This template performs RAG using Chroma and OpenAI.

##  Chroma

[Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) is an open-source vector database.

This template will create and add documents to the vector database in `chain.py`.

These documents can be loaded from [many sources](https://python.langchain.com/docs/integrations/document_loaders).

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

## Adding the template

Create your LangServe app:
```
langchain serve new my-app
cd my-app
```

Add template:
```
langchain serve add rag-chroma
```

Start server:
```
langchain start
```

See Jupyter notebook `rag_chroma` for various way to connect to the template.
