
# extraction-anthropic-functions

This template enables [Anthropic function calling](https://python.langchain.com/docs/integrations/chat/anthropic_functions). 

This can be used for various tasks, such as extraction or tagging.

The function output schema can be set in `chain.py`. 

## Environment Setup

Set the `ANTHROPIC_API_KEY` environment variable to access the Anthropic models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package extraction-anthropic-functions
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add extraction-anthropic-functions
```

And add the following code to your `server.py` file:
```python
from extraction_anthropic_functions import chain as extraction_anthropic_functions_chain

add_routes(app, extraction_anthropic_functions_chain, path="/extraction-anthropic-functions")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
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
We can access the playground at [http://127.0.0.1:8000/extraction-anthropic-functions/playground](http://127.0.0.1:8000/extraction-anthropic-functions/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/extraction-anthropic-functions")
```

By default, the package will extract the title and author of papers from the information you specify in `chain.py`. This template will use `Claude2` by default. 

---
