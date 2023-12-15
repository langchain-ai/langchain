
# rag-chroma-multi-modal-multi-vector

Presentations (slide decks, etc) contain visual content that challenges conventional RAG.

Multi-modal LLMs unlock new ways to build apps over visual content like presentations.
 
This template performs multi-modal RAG using Chroma with the multi-vector retriever (see [blog](https://blog.langchain.dev/multi-modal-rag-template/)):

* Extracts the slides as images
* Uses GPT-4V to summarize each image
* Embeds the image summaries with a link to the original images
* Retrieves relevant image based on similarity between the image summary and the user input
* Finally pass those images to GPT-4V for answer synthesis

## Storage

We will use Upstash to store the images, which offers Redis with a REST API.

Simply login [here](https://upstash.com/) and create a database.

This will give you a REST API with:

* UPSTASH_URL
* UPSTASH_TOKEN

Set `UPSTASH_URL` and `UPSTASH_TOKEN` as environment variables to access your database.

We will use Chroma to store and index the image summaries, which will be created locally in the template directory.

## Input

Supply a slide deck as pdf in the `/docs` directory. 

Create your vectorstore (Chroma) and populae Upstash with: 

```
poetry install
python ingest.py
```

## LLM

The app will retrieve images using multi-modal embeddings, and pass them to GPT-4V.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI GPT-4V.

Set `UPSTASH_URL` and `UPSTASH_TOKEN` as environment variables to access your database.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-chroma-multi-modal-multi-vector
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-chroma-multi-modal-multi-vector
```

And add the following code to your `server.py` file:
```python
from rag_chroma_multi_modal_multi_vector import chain as rag_chroma_multi_modal_chain_mv

add_routes(app, rag_chroma_multi_modal_chain_mv, path="/rag-chroma-multi-modal-multi-vector")
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
We can access the playground at [http://127.0.0.1:8000/rag-chroma-multi-modal-multi-vector/playground](http://127.0.0.1:8000/rag-chroma-multi-modal-multi-vector/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-chroma-multi-modal-multi-vector")
```