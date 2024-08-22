
# rag-jaguardb

This template performs RAG using JaguarDB and OpenAI.

## Environment Setup

You should export two environment variables, one being your Jaguar URI, the other being your OpenAI API KEY.
If you do not have JaguarDB set up, see the `Setup Jaguar` section at the bottom for instructions on how to do so.

```shell
export JAGUAR_API_KEY=...
export OPENAI_API_KEY=...
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-jaguardb
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-jagaurdb
```

And add the following code to your `server.py` file:
```python
from rag_jaguardb import chain as rag_jaguardb

add_routes(app, rag_jaguardb_chain, path="/rag-jaguardb")
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
We can access the playground at [http://127.0.0.1:8000/rag-jaguardb/playground](http://127.0.0.1:8000/rag-jaguardb/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-jaguardb")
```

## JaguarDB Setup

To utilize JaguarDB, you can use docker pull and docker run commands to quickly setup JaguarDB. 

```shell
docker pull jaguardb/jaguardb 
docker run -d -p 8888:8888 --name jaguardb jaguardb/jaguardb
```

To launch the JaguarDB client terminal to interact with JaguarDB server: 

```shell 
docker exec -it jaguardb /home/jaguar/jaguar/bin/jag
```

Another option is to download an already-built binary package of JaguarDB on Linux, and deploy the database on a single node or in a cluster of nodes. The streamlined process enables you to quickly start using JaguarDB and leverage its powerful features and functionalities. [here](http://www.jaguardb.com/download.html).   