# Conversational RAG 

This template performs [conversational](https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain) [retrieval](https://python.langchain.com/docs/use_cases/question_answering/), which is one of the most popular LLM use-cases.

It passes both a conversation history and retrieved documents into an LLM for synthesis.

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

##  Chroma

[Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) is an open-source vector database.

This template will create and add documents to the vector database in `chain.py`.

By default, this will load a popular blog post on agents.

However, you can choose from a large number of document loaders [here](https://python.langchain.com/docs/integrations/document_loaders).

## Adding the template

Create your LangServe app:
```
langchain serve new my-app
cd my-app
```

Add template:
```
langchain serve add rag-conversation
```

Start server:
```
langchain start
```

See Jupyter notebook `rag-conversation` for various way to connect to the template.
