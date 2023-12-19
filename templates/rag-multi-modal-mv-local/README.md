
# rag-multi-modal-mv-local

Multi-modal LLMs unlock new ways to build apps over visual content like photos.
 
This template performs multi-modal RAG over a set of images.

It first indexes the images and allows a user to asks questions about them.

For each question, it will retrieve the relevant image and pass the image to a multi-modal LLM to generate the answer.

To do this, it uses the multi-vector retriever (see [blog](https://blog.langchain.dev/multi-modal-rag-template/)):

* Given a set of images
* It uses a local multi-modal LLM ([bakllava](https://ollama.ai/library/bakllava)) to summarize each image
* Embeds the image summaries with a link to the original images
* Given a user question, it will relevant image(s) based on similarity between the image summary and user input
* It will pass those images to bakllava for answer synthesis

All these steps will be done using local, open-source LLMs.

## LLM

We will use [Ollama](https://python.langchain.com/docs/integrations/chat/ollama#multi-modal) for generating image summaries and final image QA.

Download the latest version of Ollama: https://ollama.ai/

Pull the an open source multi-modal LLM: e.g., https://ollama.ai/library/bakllava

```
ollama pull baklava
```

The app is by default configured for `baklava`. But you can change this in `chain.py` and `ingest.py` for different downloaded models.

## Input

Supply a set of images the `/docs` directory and run:

```
poetry install
python ingest.py
```

This will create a vectorstore (Chroma) of image summaries that are embedded using [Ollama embeddings](https://python.langchain.com/docs/integrations/text_embedding/ollama). 

Each image summaries is linked to the raw images, which is stored in a [local file store](https://python.langchain.com/docs/integrations/stores/file_system).

This allows us to retrieve images based on natural langugae text summaries.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-multi-modal-mv-local
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-multi-modal-mv-local
```

And add the following code to your `server.py` file:
```python
from rag_multi_modal_mv_local import chain as rag_multi_modal_mv_local_chain

add_routes(app, rag_multi_modal_mv_local_chain, path="/rag-multi-modal-mv-local")
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
We can access the playground at [http://127.0.0.1:8000/rag-multi-modal-mv-local/playground](http://127.0.0.1:8000/rag-multi-modal-mv-local/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-multi-modal-mv-local")
```