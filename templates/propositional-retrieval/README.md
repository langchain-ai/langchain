# propositional-retrieval

This template demonstrates the multi-vector indexing strategy proposed by Chen, et. al.'s [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648). The prompt, which you can [try out on the hub](https://smith.langchain.com/hub/wfh/proposal-indexing), directs an LLM to generate de-contextualized "propositions" which can be vectorized to increase the retrieval accuracy. You can see the full definition in `proposal_chain.py`.

![Diagram illustrating the multi-vector indexing strategy for information retrieval, showing the process from Wikipedia data through a Proposition-izer to FactoidWiki, and the retrieval of information units for a QA model.](https://github.com/langchain-ai/langchain/raw/master/templates/propositional-retrieval/_images/retriever_diagram.png "Retriever Diagram")

## Storage

For this demo, we index a simple academic paper using the RecursiveUrlLoader, and store all retriever information locally (using chroma and a bytestore stored on the local filesystem). You can modify the storage layer in `storage.py`.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access `gpt-3.5` and the OpenAI Embeddings classes.

## Indexing

Create the index by running the following:

```python
poetry install
poetry run python propositional_retrieval/ingest.py
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package propositional-retrieval
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add propositional-retrieval
```

And add the following code to your `server.py` file:

```python
from propositional_retrieval import chain

add_routes(app, chain, path="/propositional-retrieval")
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
We can access the playground at [http://127.0.0.1:8000/propositional-retrieval/playground](http://127.0.0.1:8000/propositional-retrieval/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/propositional-retrieval")
```
