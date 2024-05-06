# retrieval-agent-fireworks

This package uses open source models hosted on FireworksAI to do retrieval using an agent architecture. By default, this does retrieval over Arxiv.

We will use `Mixtral8x7b-instruct-v0.1`, which is shown in this blog to yield reasonable
results with function calling even though it is not fine tuned for this task: https://huggingface.co/blog/open-source-llms-as-agents


## Environment Setup

There are various great ways to run OSS models. We will use FireworksAI as an easy way to run the models. See [here](https://python.langchain.com/docs/integrations/providers/fireworks) for more information.

Set the `FIREWORKS_API_KEY` environment variable to access Fireworks.


## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package retrieval-agent-fireworks
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add retrieval-agent-fireworks
```

And add the following code to your `server.py` file:
```python
from retrieval_agent_fireworks import chain as retrieval_agent_fireworks_chain

add_routes(app, retrieval_agent_fireworks_chain, path="/retrieval-agent-fireworks")
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
We can access the playground at [http://127.0.0.1:8000/retrieval-agent-fireworks/playground](http://127.0.0.1:8000/retrieval-agent-fireworks/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/retrieval-agent-fireworks")
```