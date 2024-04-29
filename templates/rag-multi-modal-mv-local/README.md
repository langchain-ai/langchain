
# rag-multi-modal-mv-local

Visual search is a famililar application to many with iPhones or Android devices. It allows user to search photos using natural language.
  
With the release of open source, multi-modal LLMs it's possible to build this kind of application for yourself for your own private photo collection.

This template demonstrates how to perform private visual search and question-answering over a collection of your photos.

It uses an open source multi-modal LLM of your choice to create image summaries for each photos, embeds the summaries, and stores them in Chroma.
 
Given a question, relevat photos are retrieved and passed to the multi-modal LLM for answer synthesis.

![Diagram illustrating the visual search process with food pictures, captioning, a database, a question input, and the synthesis of an answer using a multi-modal LLM.](https://github.com/langchain-ai/langchain/assets/122662504/cd9b3d82-9b06-4a39-8490-7482466baf43 "Visual Search Process Diagram")

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
* Given a user question, it will relevant image(s) based on similarity between the image summary and user input (using Ollama embeddings)
* It will pass those images to bakllava for answer synthesis

By default, this will use [LocalFileStore](https://python.langchain.com/docs/integrations/stores/file_system) to store images and Chroma to store summaries.

## LLM and Embedding Models

We will use [Ollama](https://python.langchain.com/docs/integrations/chat/ollama#multi-modal) for generating image summaries, embeddings, and the final image QA.

Download the latest version of Ollama: https://ollama.ai/

Pull an open source multi-modal LLM: e.g., https://ollama.ai/library/bakllava

Pull an open source embedding model: e.g., https://ollama.ai/library/llama2:7b

```
ollama pull bakllava
ollama pull llama2:7b
```

The app is by default configured for `bakllava`. But you can change this in `chain.py` and `ingest.py` for different downloaded models.

The app will retrieve images based on similarity between the text input and the image summary, and pass the images to `bakllava`.

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
You can sign up for LangSmith [here](https://smith.langchain.com/). 
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
