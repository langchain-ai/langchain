
# rag-multi-modal-local

Visual search is a famililar application to many with iPhones or Android devices. It allows user to search photos using natural language.
  
With the release of open source, multi-modal LLMs it's possible to build this kind of application for yourself for your own private photo collection.

This template demonstrates how to perform private visual search and question-answering over a collection of your photos.

It uses OpenCLIP embeddings to embed all of the photos and stores them in Chroma.
 
Given a question, relevant photos are retrieved and passed to an open source multi-modal LLM of your choice for answer synthesis.
 
![Diagram illustrating the visual search process with OpenCLIP embeddings and multi-modal LLM for question-answering, featuring example food pictures and a matcha soft serve answer trace.](https://github.com/langchain-ai/langchain/assets/122662504/da543b21-052c-4c43-939e-d4f882a45d75 "Visual Search Process Diagram")

## Input

Supply a set of photos in the `/docs` directory. 

By default, this template has a toy collection of 3 food pictures.

Example questions to ask can be:
```
What kind of soft serve did I have?
```

In practice, a larger corpus of images can be tested.

To create an index of the images, run:
```
poetry install
python ingest.py
```

## Storage

This template will use [OpenCLIP](https://github.com/mlfoundations/open_clip) multi-modal embeddings to embed the images.

You can select different embedding model options (see results [here](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv)).

The first time you run the app, it will automatically download the multimodal embedding model.

By default, LangChain will use an embedding model with moderate performance but lower memory requirments, `ViT-H-14`.

You can choose alternative `OpenCLIPEmbeddings` models in `rag_chroma_multi_modal/ingest.py`:
```
vectorstore_mmembd = Chroma(
    collection_name="multi-modal-rag",
    persist_directory=str(re_vectorstore_path),
    embedding_function=OpenCLIPEmbeddings(
        model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k"
    ),
)
```

## LLM

This template will use [Ollama](https://python.langchain.com/docs/integrations/chat/ollama#multi-modal).

Download the latest version of Ollama: https://ollama.ai/

Pull the an open source multi-modal LLM: e.g., https://ollama.ai/library/bakllava

```
ollama pull bakllava
```

The app is by default configured for `bakllava`. But you can change this in `chain.py` and `ingest.py` for different downloaded models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-chroma-multi-modal
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-chroma-multi-modal
```

And add the following code to your `server.py` file:
```python
from rag_chroma_multi_modal import chain as rag_chroma_multi_modal_chain

add_routes(app, rag_chroma_multi_modal_chain, path="/rag-chroma-multi-modal")
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
We can access the playground at [http://127.0.0.1:8000/rag-chroma-multi-modal/playground](http://127.0.0.1:8000/rag-chroma-multi-modal/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-chroma-multi-modal")
```
