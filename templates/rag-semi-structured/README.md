# Semi structured RAG

This package performs Retrieval Augmented Generation (RAG) on semi-structured data, such as a PDF with text and tables.

## Environment Setup

This package requires some system-level package installations. On Mac, you can install the necessary packages with the following command:

```shell
brew install tesseract poppler
```

Make sure that the `OPENAI_API_KEY` is set in order to use the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-semi-structured
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-semi-structured
```

And add the following code to your `server.py` file:
```python
TODO: __app_route_code__
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
We can access the playground at [http://127.0.0.1:8000/rag-semi-structured/playground](http://127.0.0.1:8000/rag-semi-structured/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-semi-structured")
```

For more details on how to connect to the template, refer to the Jupyter notebook `rag_semi_structured`.