#  RAG Pinecone multi query

This template performs RAG using Pinecone and OpenAI with the [multi-query retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever).

This will use an LLM to generate multiple queries from different perspectives for a given user input query. 

For each query, it retrieves a set of relevant documents and takes the unique union across all queries for answer synthesis.

##  Pinecone

This template uses Pinecone as a vectorstore and requires that `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and `PINECONE_INDEX` are set.

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

## App

Example `server.py`:
```
from fastapi import FastAPI
from langserve import add_routes
from rag_pinecone_multi_query.chain import chain

app = FastAPI()

# Edit this to add the chain you want to add
add_routes(app, chain, path="rag_pinecone_multi_query")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
```

Run:
```
python app/server.py
```

Check endpoint:
```
http://0.0.0.0:8001/docs
```

See `rag_pinecone_multi_query.ipynb` for example usage - 
```
from langserve.client import RemoteRunnable
rag_app_pinecone = RemoteRunnable('http://0.0.0.0:8001/rag_pinecone_multi_query')
rag_app_pinecone.invoke("What are the different types of agent memory")
```