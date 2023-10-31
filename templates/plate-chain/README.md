Plate-chain is a Python package that allows for intricate plate management and arrangement. It is used in conjunction with the LangChain CLI to create and add to existing projects for an enhanced user experience.

## Environment Setup

There are no environment variables required to run this package.

## Usage

To utilize plate-chain, you must have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

Creating a new LangChain project and installing plate-chain as the only package can be done with:

```shell
langchain app new my-app --package plate-chain
```

If you wish to add this to an existing project, simply run:

```shell
langchain app add plate-chain
```

Then add the following code to your `server.py` file:

```python
from plate_chain import chain as plate_chain_chain

add_routes(app, plate_chain_chain, path="/plate-chain")
```

(Optional) For configuring LangSmith, which helps trace, monitor and debug LangChain applications, use the following code:

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you're in this directory, you can start a LangServe instance directly by:

```shell
langchain serve
```

This starts the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

All templates can be viewed at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Access the playground at [http://127.0.0.1:8000/plate-chain/playground](http://127.0.0.1:8000/plate-chain/playground)  

You can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/plate-chain")
```