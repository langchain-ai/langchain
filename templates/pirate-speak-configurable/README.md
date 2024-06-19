# pirate-speak-configurable

This template converts user input into pirate speak. It shows how you can allow
`configurable_alternatives` in the Runnable, allowing you to select from 
OpenAI, Anthropic, or Cohere as your LLM Provider in the playground (or via API).

## Environment Setup

Set the following environment variables to access all 3 configurable alternative
model providers:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `COHERE_API_KEY`

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package pirate-speak-configurable
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add pirate-speak-configurable
```

And add the following code to your `server.py` file:
```python
from pirate_speak_configurable import chain as pirate_speak_configurable_chain

add_routes(app, pirate_speak_configurable_chain, path="/pirate-speak-configurable")
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
We can access the playground at [http://127.0.0.1:8000/pirate-speak-configurable/playground](http://127.0.0.1:8000/pirate-speak-configurable/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/pirate-speak-configurable")
```