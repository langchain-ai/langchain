
# llama2-functions

This template performs extraction of structured data from unstructured data using a [LLaMA2 model that supports a specified JSON output schema](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). 

The extraction schema can be set in `chain.py`.

## Environment Setup

This will use a [LLaMA2-13b model hosted by Replicate](https://replicate.com/andreasjansson/llama-2-13b-chat-gguf/versions).

Ensure that `REPLICATE_API_TOKEN` is set in your environment.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package llama2-functions
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add llama2-functions
```

And add the following code to your `server.py` file:
```python
from llama2_functions import chain as llama2_functions_chain

add_routes(app, llama2_functions_chain, path="/llama2-functions")
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
We can access the playground at [http://127.0.0.1:8000/llama2-functions/playground](http://127.0.0.1:8000/llama2-functions/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/llama2-functions")
```
