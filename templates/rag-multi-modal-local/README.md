
# rag-chroma-multi-modal

Multi-modal LLMs unlock new ways to build apps over visual content like photos.
 
This template performs multi-modal RAG over a set of images.

It first indexes the images and allows a user to asks questions about them.

For each question, it will retrieve the relevant image and pass the image to a multi-modal LLM to generate the answer.

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

Supply a set of images the `/docs` directory. 

This will create a vectorstore (Chroma) of image summaries that are embedded using OpenCLIP embeddings.

```
poetry install
python ingest.py
```

## Embeddings

This template will use [OpenCLIP](https://github.com/mlfoundations/open_clip) multi-modal embeddings.

You can select different options (see results [here](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv)).

The first time you run the app, it will automatically download the multimodal embedding model.

By default, LangChain will use an embedding model with reasonably strong performance, `ViT-H-14`.

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
We can access the playground at [http://127.0.0.1:8000/rag-chroma-multi-modal/playground](http://127.0.0.1:8000/rag-chroma-multi-modal/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-chroma-multi-modal")
```