# RAG with Supabase

> [Supabase](https://supabase.com/docs) is an open-source Firebase alternative. It is built on top of [PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL), a free and open-source relational database management system (RDBMS) and uses [pgvector](https://github.com/pgvector/pgvector) to store embeddings within your tables.

Use this package to host a retrieval augment generation (RAG) API using LangServe + Supabase.

## Install Package

From within your `langservehub` project run:

```shell
poetry run poe add rag-supabase
```

## Setup Supabase Database

Use these steps to setup your Supabase database if you haven't already.

1. Head over to https://database.new to provision your Supabase database.
2. In the studio, jump to the [SQL editor](https://supabase.com/dashboard/project/_/sql/new) and run the following script to enable `pgvector` and setup your database as a vector store:

   ```sql
   -- Enable the pgvector extension to work with embedding vectors
   create extension if not exists vector;

   -- Create a table to store your documents
   create table
     documents (
       id uuid primary key,
       content text, -- corresponds to Document.pageContent
       metadata jsonb, -- corresponds to Document.metadata
       embedding vector (1536) -- 1536 works for OpenAI embeddings, change as needed
     );

   -- Create a function to search for documents
   create function match_documents (
     query_embedding vector (1536),
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

Since we are using [`SupabaseVectorStore`](https://python.langchain.com/docs/integrations/vectorstores/supabase) and [`OpenAIEmbeddings`](https://python.langchain.com/docs/integrations/text_embedding/openai), we need to load their API keys.

Create a `.env` file in the root of your project:

_.env_

```shell
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
OPENAI_API_KEY=
```

To find your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`, head to your Supabase project's [API settings](https://supabase.com/dashboard/project/_/settings/api).

- `SUPABASE_URL` corresponds to the Project URL
- `SUPABASE_SERVICE_KEY` corresponds to the `service_role` API key

To get your `OPENAI_API_KEY`, navigate to [API keys](https://platform.openai.com/account/api-keys) on your OpenAI account and create a new secret key.

Add this file to your `.gitignore` if it isn't already there (so that we don't commit secrets):

_.gitignore_

```
.env
```

Install [`python-dotenv`](https://github.com/theskumar/python-dotenv) which we will use to load the environment variables into the app:

```shell
poetry add python-dotenv
```

Finally, call `load_dotenv()` in `server.py`.

_app/server.py_

```python
from dotenv import load_dotenv

load_dotenv()
```
