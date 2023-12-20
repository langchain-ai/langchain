
# rag-multi-modal-mv-local

Multi-modal LLMs enable text-to-image retrieval and question-answering over images. 

You can ask questions in natural language about a collection of photos, retrieve relevant ones, and have a multi-modal LLM answer questions about the retrieved images.

This template performs text-to-image retrieval for question-answering about a slide deck, which often contains visual elements that are not captured in standard RAG.

This will use Ollama to run a local multi-modal LLM for image captioning and answer synthesis.

## Input

Supply a set of photos in the `/docs` directory. 

By default, this template has a toy collection of 3 food pictures.

The app will look up and summarize photos based upon provided keywords or questions:
```
What kind of ice cream did I have?
```

In practice, a larger corpus of images can be tested.

To create an index of the images, run:
```
poetry install
python ingest.py
```

## Storage

Here is the process the template will use to create an index of the slides (see [blog](https://blog.langchain.dev/multi-modal-rag-template/)):

* Given a set of images
* It uses a local multi-modal LLM ([bakllava](https://ollama.ai/library/bakllava)) to summarize each image
* Embeds the image summaries with a link to the original images
* Given a user question, it will relevant image(s) based on similarity between the image summary and user input (using GPT4All embeddings)
* It will pass those images to bakllava for answer synthesis

By default, this will use [LocalFileStore](https://python.langchain.com/docs/integrations/stores/file_system) to store images and Chroma to store summaries.

## LLM

We will use [Ollama](https://python.langchain.com/docs/integrations/chat/ollama#multi-modal) for generating image summaries and final image QA.

Download the latest version of Ollama: https://ollama.ai/

Pull the an open source multi-modal LLM: e.g., https://ollama.ai/library/bakllava

```
ollama pull baklava
```

The app is by default configured for `baklava`. But you can change this in `chain.py` and `ingest.py` for different downloaded models.

The app will retrieve images based on similarity between the text input and the image summary, and pass the images to `baklava`.

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