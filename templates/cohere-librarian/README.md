
# cohere-librarian

This template turns Cohere into a librarian.

It demonstrates the use of a router to switch between chains that can handle different things: a vector database with Cohere embeddings; a chat bot that has a prompt with some information about the library; and finally a RAG chatbot that has access to the internet.

For a fuller demo of the book recomendation, consider replacing books_with_blurbs.csv with a larger sample from the following dataset: https://www.kaggle.com/datasets/jdobrow/57000-books-with-metadata-and-blurbs/ .

## Environment Setup

Set the `COHERE_API_KEY` environment variable to access the Cohere models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package cohere-librarian
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add cohere-librarian
```

And add the following code to your `server.py` file:
```python
from cohere_librarian.chain import chain as cohere_librarian_chain

add_routes(app, cohere_librarian_chain, path="/cohere-librarian")
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

We can see all templates at [http://localhost:8000/docs](http://localhost:8000/docs)
We can access the playground at [http://localhost:8000/cohere-librarian/playground](http://localhost:8000/cohere-librarian/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/cohere-librarian")
```
