
# self-query-qdrant

This template performs [self-querying](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/) 
using Qdrant and OpenAI. By default, it uses an artificial dataset of 10 documents, but you can replace it with your own dataset.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

Set the `QDRANT_URL` to the URL of your Qdrant instance. If you use [Qdrant Cloud](https://cloud.qdrant.io)
you have to set the `QDRANT_API_KEY` environment variable as well. If you do not set any of them,
the template will try to connect a local Qdrant instance at `http://localhost:6333`.

```shell
export QDRANT_URL=
export QDRANT_API_KEY=

export OPENAI_API_KEY=
```

## Usage

To use this package, install the LangChain CLI first:

```shell
pip install -U "langchain-cli[serve]"
```

Create a new LangChain project and install this package as the only one:

```shell
langchain app new my-app --package self-query-qdrant
```

To add this to an existing project, run:

```shell
langchain app add self-query-qdrant
```

### Defaults

Before you launch the server, you need to create a Qdrant collection and index the documents.
It can be done by running the following command:

```python
from self_query_qdrant.chain import initialize

initialize()
```

Add the following code to your `app/server.py` file:

```python
from self_query_qdrant.chain import chain

add_routes(app, chain, path="/self-query-qdrant")
```

The default dataset consists 10 documents about dishes, along with their price and restaurant information.
You can find the documents in the `packages/self-query-qdrant/self_query_qdrant/defaults.py` file.
Here is one of the documents:

```python
from langchain_core.documents import Document

Document(
    page_content="Spaghetti with meatballs and tomato sauce",
    metadata={
        "price": 12.99,
        "restaurant": {
            "name": "Olive Garden",
            "location": ["New York", "Chicago", "Los Angeles"],
        },
    },
)
```

The self-querying allows performing semantic search over the documents, with some additional filtering
based on the metadata. For example, you can search for the dishes that cost less than $15 and are served in New York.

### Customization

All the examples above assume that you want to launch the template with just the defaults.
If you want to customize the template, you can do it by passing the parameters to the `create_chain` function
in the `app/server.py` file:

```python
from langchain_community.llms import Cohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo

from self_query_qdrant.chain import create_chain

chain = create_chain(
    llm=Cohere(),
    embeddings=HuggingFaceEmbeddings(),
    document_contents="Descriptions of cats, along with their names and breeds.",
    metadata_field_info=[
        AttributeInfo(name="name", description="Name of the cat", type="string"),
        AttributeInfo(name="breed", description="Cat's breed", type="string"),
    ],
    collection_name="cats",
)
```

The same goes for the `initialize` function that creates a Qdrant collection and indexes the documents:

```python
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from self_query_qdrant.chain import initialize

initialize(
    embeddings=HuggingFaceEmbeddings(),
    collection_name="cats",
    documents=[
        Document(
            page_content="A mean lazy old cat who destroys furniture and eats lasagna",
            metadata={"name": "Garfield", "breed": "Tabby"},
        ),
        ...
    ]
)
```

The template is flexible and might be used for different sets of documents easily.

### LangSmith

(Optional) If you have access to LangSmith, configure it to help trace, monitor and debug LangChain applications. If you don't have access, skip this section.

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

### Local Server

This will start the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

You can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Access the playground at [http://127.0.0.1:8000/self-query-qdrant/playground](http://127.0.0.1:8000/self-query-qdrant/playground)

Access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/self-query-qdrant")
```
