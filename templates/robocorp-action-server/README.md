# Langchain - Robocorp Action Server

This template enables using [Robocorp Action Server](https://github.com/robocorp/robocorp) served actions as tools for an Agent.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package robocorp-action-server
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add robocorp-action-server
```

And add the following code to your `server.py` file:

```python
from robocorp_action_server import agent_executor as action_server_chain

add_routes(app, action_server_chain, path="/robocorp-action-server")
```

### Running the Action Server

To run the Action Server, you need to have the Robocorp Action Server installed

```bash
pip install -U robocorp-action-server
```

Then you can run the Action Server with:

```bash
action-server new
cd ./your-project-name
action-server start
```

### Configure LangSmith (Optional)

LangSmith will help us trace, monitor and debug LangChain applications.
You can sign up for LangSmith [here](https://smith.langchain.com/).
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

### Start LangServe instance

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/robocorp-action-server/playground](http://127.0.0.1:8000/robocorp-action-server/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/robocorp-action-server")
```
