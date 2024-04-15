# research-assistant

This template implements a version of  
[GPT Researcher](https://github.com/assafelovic/gpt-researcher) that you can use
as a starting point for a research agent.

## Environment Setup

The default template relies on ChatOpenAI and DuckDuckGo, so you will need the 
following environment variable:

- `OPENAI_API_KEY`

And to use the Tavily LLM-optimized search engine, you will need:

- `TAVILY_API_KEY`

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package research-assistant
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add research-assistant
```

And add the following code to your `server.py` file:
```python
from research_assistant import chain as research_assistant_chain

add_routes(app, research_assistant_chain, path="/research-assistant")
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
We can access the playground at [http://127.0.0.1:8000/research-assistant/playground](http://127.0.0.1:8000/research-assistant/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/research-assistant")
```