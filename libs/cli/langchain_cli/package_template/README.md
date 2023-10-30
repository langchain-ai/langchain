# __package_name_last__

TODO: What does this package do

## Environment Setup

TODO: What environment variables need to be set (if any)

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package __package_name__
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add __package_name__
```

And add the following code to your `server.py` file:
```python

```

To see and debug your chains with [LangSmith](https://docs.smith.langchain.com/), set the following environment variables. You can get an API key [here](https://smith.langchain.com/settings):
```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/__package_name__/playground](http://127.0.0.1:8000/__package_name__/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/__package_name__")
```