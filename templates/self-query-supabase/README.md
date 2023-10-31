
# self-query-supabase

This package is designed to host a LangServe API that can self-query Supabase. It allows you to use natural language to generate a structured query against the database. [Supabase](https://supabase.com/docs) is an open-source alternative to Firebase, built on top of [PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL). It uses [pgvector](https://github.com/pgvector/pgvector) to store embeddings within your tables.

## Environment Setup

You need to load the API keys for [`SupabaseVectorStore`](https://python.langchain.com/docs/integrations/vectorstores/supabase) and [`OpenAIEmbeddings`](https://python.langchain.com/docs/integrations/text_embedding/openai) as environment variables. Create a `.env` file in your project's root directory:

_.env_

```shell
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
OPENAI_API_KEY=
```

To find your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`, head to your Supabase project's [API settings](https://supabase.com/dashboard/project/_/settings/api). `SUPABASE_URL` corresponds to the Project URL and `SUPABASE_SERVICE_KEY` corresponds to the `service_role` API key. To get your `OPENAI_API_KEY`, navigate to [API keys](https://platform.openai.com/account/api-keys) on your OpenAI account and create a new secret key. Add this file to your `.gitignore` to prevent committing secrets:

_.gitignore_

```
.env
```

## Usage

To use this package, install the LangChain CLI first:

```shell
pip install -U "langchain-cli[serve]"
```

Create a new LangChain project and install this package as the only one:

```shell
langchain app new my-app --package self-query-supabase
```

To add this to an existing project, run:

```shell
langchain app add self-query-supabase
```

Add the following code to your `server.py` file:
```python
from self_query_supabase import chain as self_query_supabase_chain

add_routes(app, self_query_supabase_chain, path="/self-query-supabase")
```

(Optional) If you have access to LangSmith, configure it to help trace, monitor and debug LangChain applications. If you don't have access, skip this section.

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

You can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Access the playground at [http://127.0.0.1:8000/self-query-supabase/playground](http://127.0.0.1:8000/self-query-supabase/playground)

Access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/self-query-supabase")
```

TODO: Instructions to set up the Supabase database and install the package.
