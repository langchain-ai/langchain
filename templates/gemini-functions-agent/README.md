
# gemini-functions-agent

This template creates an agent that uses Google Gemini function calling to communicate its decisions on what actions to take. 

This example creates an agent that can optionally look up information on the internet using Tavily's search engine.

[See an example LangSmith trace here](https://smith.langchain.com/public/0ebf1bd6-b048-4019-b4de-25efe8d3d18c/r)

## Environment Setup

The following environment variables need to be set:

Set the `TAVILY_API_KEY` environment variable to access Tavily

Set the `GOOGLE_API_KEY` environment variable to access the Google Gemini APIs.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package gemini-functions-agent
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add gemini-functions-agent
```

And add the following code to your `server.py` file:
```python
from gemini_functions_agent import agent_executor as gemini_functions_agent_chain

add_routes(app, gemini_functions_agent_chain, path="/openai-functions-agent")
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
We can access the playground at [http://127.0.0.1:8000/gemini-functions-agent/playground](http://127.0.0.1:8000/gemini-functions-agent/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/gemini-functions-agent")
```