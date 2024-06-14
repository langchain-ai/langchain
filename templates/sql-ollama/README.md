# sql-ollama

This template enables a user to interact with a SQL database using natural language. 

It uses [Zephyr-7b](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) via [Ollama](https://ollama.ai/library/zephyr) to run inference locally on a Mac laptop.

## Environment Setup

Before using this template, you need to set up Ollama and SQL database.

1. Follow instructions [here](https://python.langchain.com/docs/integrations/chat/ollama) to download Ollama.

2. Download your LLM of interest:

    * This package uses `zephyr`: `ollama pull zephyr`
    * You can choose from many LLMs [here](https://ollama.ai/library)

3. This package includes an example DB of 2023 NBA rosters. You can see instructions to build this DB [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package sql-ollama
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add sql-ollama
```

And add the following code to your `server.py` file:

```python
from sql_ollama import chain as sql_ollama_chain

add_routes(app, sql_ollama_chain, path="/sql-ollama")
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
We can access the playground at [http://127.0.0.1:8000/sql-ollama/playground](http://127.0.0.1:8000/sql-ollama/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/sql-ollama")
```