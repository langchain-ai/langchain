
# rag-vectara-selfquery

This template performs [self-querying](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/) 
using Vectara and OpenAI. By default, it uses an artificial dataset of 6 documents, but you can replace it with your own dataset.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

Also, ensure the following environment variables are set:
* `VECTARA_CUSTOMER_ID`
* `VECTARA_CORPUS_ID`
* `VECTARA_API_KEY`

Please make sure that your corpus (pointed to be `VECTARA_CORPUS_ID`) has the following 
[metadata filter](https://docs.vectara.com/docs/learn/metadata-search-filtering/filter-overview) attributes defined, 
to match our example documents:
* `genre` of type string
* `year` of type integer
* `director` of type string
* `rating` of type float

## Usage - with default example

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-vectara-selfquery
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-vectara-selfquery
```

Add the following code to your `server.py` file:
```python
from rag_vectara import chain as rag_vectara_chain

add_routes(app, rag_vectara_chain, path="/rag-vectara-selfquery")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "vectara-demo"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/rag-vectara-selfquery/playground](http://127.0.0.1:8000/rag-vectara-selfquery/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-vectara-selfquery")
```

## Usage - with your own data

The example above uses the example documents and schema defined in the `packages/rag-vectara-selfquery/rag_vectara_selfquery/defaults.py` file.
To use it with your own data, please follow these steps.

First, make sure to setup your Vectara corpus with the correct 
[metadata filter](https://docs.vectara.com/docs/learn/metadata-search-filtering/filter-overview) attributes defined,
to match your dataset. Specifically, for self-query you need to define a filtering attribute if you want
self-query to automaticaly filter by that attribute. 

Now create your chain by passing the parameters to the `create_chain` function in the `app/server.py` file:

```python
from langchain.llms import Cohere
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo

from rag_vectara_selfquery.chain import create_chain

chain = create_chain(
    llm=OpenAI(temperature=0),
    document_contents="Descriptions of products, along with their prices and categories.",
    metadata_field_info=[
        AttributeInfo(name="name", description="Name of the product", type="string"),
        AttributeInfo(name="price", description="product price", type="float"),
        AttributeInfo(name="category", description="product category", type="string")
    ],
)
```

And now, as with the default example:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/rag-vectara-selfquery/playground](http://127.0.0.1:8000/rag-vectara-selfquery/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-vectara-selfquery")
```




