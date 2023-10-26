#  RAG Pinecone multi query

This template performs RAG using Pinecone and OpenAI with the [multi-query retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever).

This will use an LLM to generate multiple queries from different perspectives for a given user input query. 

For each query, it retrieves a set of relevant documents and takes the unique union across all queries for answer synthesis.

##  Pinecone

This template uses Pinecone as a vectorstore and requires that `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and `PINECONE_INDEX` are set.

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

