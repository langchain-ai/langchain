
# rag-chroma-multi-modal-multi-vector

Multi-modal LLMs enable visual assistants that can perform question-answering about images. 

This template create a visual assistant for slide decks, which often contain visuals such as graphs or figures.

It uses GPT-4V to create image summaries for each slide, embeds the summaries, and stores them in Chroma.
 
Given a question, relevat slides are retrieved and passed to GPT-4V for answer synthesis.

![Diagram illustrating the multi-modal LLM process with a slide deck, captioning, storage, question input, and answer synthesis with year-over-year growth percentages.](https://github.com/langchain-ai/langchain/assets/122662504/5277ef6b-d637-43c7-8dc1-9b1567470503 "Multi-modal LLM Process Diagram")

## Input

Supply a slide deck as pdf in the `/docs` directory. 

By default, this template has a slide deck about Q3 earnings from DataDog, a public techologyy company.

Example questions to ask can be:
```
How many customers does Datadog have?
What is Datadog platform % Y/Y growth in FY20, FY21, and FY22?
```

To create an index of the slide deck, run:
```
poetry install
python ingest.py
```

## Storage

Here is the process the template will use to create an index of the slides (see [blog](https://blog.langchain.dev/multi-modal-rag-template/)):

* Extract the slides as a collection of images
* Use GPT-4V to summarize each image
* Embed the image summaries using text embeddings with a link to the original images
* Retrieve relevant image based on similarity between the image summary and the user input question
* Pass those images to GPT-4V for answer synthesis

By default, this will use [LocalFileStore](https://python.langchain.com/docs/integrations/stores/file_system) to store images and Chroma to store summaries.

For production, it may be desirable to use a remote option such as Redis.

You can set the `local_file_store` flag in `chain.py` and `ingest.py` to switch between the two options.

For Redis, the template will use [UpstashRedisByteStore](https://python.langchain.com/docs/integrations/stores/upstash_redis).

We will use Upstash to store the images, which offers Redis with a REST API.

Simply login [here](https://upstash.com/) and create a database.

This will give you a REST API with:

* `UPSTASH_URL`
* `UPSTASH_TOKEN`
 
Set `UPSTASH_URL` and `UPSTASH_TOKEN` as environment variables to access your database.

We will use Chroma to store and index the image summaries, which will be created locally in the template directory.

## LLM

The app will retrieve images based on similarity between the text input and the image summary, and pass the images to GPT-4V.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI GPT-4V.

Set `UPSTASH_URL` and `UPSTASH_TOKEN` as environment variables to access your database if you use `UpstashRedisByteStore`.

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
We can access the playground at [http://127.0.0.1:8000/rag-chroma-multi-modal-multi-vector/playground](http://127.0.0.1:8000/rag-chroma-multi-modal-multi-vector/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-chroma-multi-modal-multi-vector")
```
