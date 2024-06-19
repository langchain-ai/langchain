
# rag_lantern

This template performs RAG with Lantern.

[Lantern](https://lantern.dev) is an open-source vector database built on top of [PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL). It enables vector search and embedding generation inside your database.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

To get your `OPENAI_API_KEY`, navigate to [API keys](https://platform.openai.com/account/api-keys) on your OpenAI account and create a new secret key.

To find your `LANTERN_URL` and `LANTERN_SERVICE_KEY`, head to your Lantern project's [API settings](https://lantern.dev/dashboard/project/_/settings/api). 

- `LANTERN_URL` corresponds to the Project URL
- `LANTERN_SERVICE_KEY` corresponds to the `service_role` API key


```shell
export LANTERN_URL=
export LANTERN_SERVICE_KEY=
export OPENAI_API_KEY=
```

## Setup Lantern Database

Use these steps to setup your Lantern database if you haven't already.

1. Head to [https://lantern.dev](https://lantern.dev) to create your Lantern database.
2. In your favorite SQL client, jump to the SQL editor and run the following script to setup your database as a vector store:

   ```sql
   -- Create a table to store your documents
   create table
     documents (
       id uuid primary key,
       content text, -- corresponds to Document.pageContent
       metadata jsonb, -- corresponds to Document.metadata
       embedding REAL[1536] -- 1536 works for OpenAI embeddings, change as needed
     );

   -- Create a function to search for documents
   create function match_documents (
     query_embedding REAL[1536],
     filter jsonb default '{}'
   ) returns table (
     id uuid,
     content text,
     metadata jsonb,
     similarity float
   ) language plpgsql as $$
   #variable_conflict use_column
   begin
     return query
     select
       id,
       content,
       metadata,
       1 - (documents.embedding <=> query_embedding) as similarity
     from documents
     where metadata @> filter
     order by documents.embedding <=> query_embedding;
   end;
   $$;
   ```

## Setup Environment Variables

Since we are using [`Lantern`](https://python.langchain.com/docs/integrations/vectorstores/lantern) and [`OpenAIEmbeddings`](https://python.langchain.com/docs/integrations/text_embedding/openai), we need to load their API keys.

## Usage

First, install the LangChain CLI:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-lantern
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-lantern
```

And add the following code to your `server.py` file:

```python
from rag_lantern.chain import chain as rag_lantern_chain

add_routes(app, rag_lantern_chain, path="/rag-lantern")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/rag-lantern/playground](http://127.0.0.1:8000/rag-lantern/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-lantern")
```
