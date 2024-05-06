
# nvidia-rag-canonical

This template performs RAG using Milvus Vector Store and NVIDIA Models (Embedding and Chat).

## Environment Setup

You should export your NVIDIA API Key as an environment variable.
If you do not have an NVIDIA API Key, you can create one by following these steps:
1. Create a free account with the [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) service, which hosts AI solution catalogs, containers, models, etc.
2. Navigate to `Catalog > AI Foundation Models > (Model with API endpoint)`.
3. Select the `API` option and click `Generate Key`.
4. Save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.

```shell
export NVIDIA_API_KEY=...
```

For instructions on hosting the Milvus Vector Store, refer to the section at the bottom.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To use the NVIDIA models, install the Langchain NVIDIA AI Endpoints package:
```shell
pip install -U langchain_nvidia_aiplay
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package nvidia-rag-canonical
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add nvidia-rag-canonical
```

And add the following code to your `server.py` file:
```python
from nvidia_rag_canonical import chain as nvidia_rag_canonical_chain

add_routes(app, nvidia_rag_canonical_chain, path="/nvidia-rag-canonical")
```

If you want to set up an ingestion pipeline, you can add the following code to your `server.py` file:
```python
from nvidia_rag_canonical import ingest as nvidia_rag_ingest

add_routes(app, nvidia_rag_ingest, path="/nvidia-rag-ingest")
```
Note that for files ingested by the ingestion API, the server will need to be restarted for the newly ingested files to be accessible by the retriever.

(Optional) Let's now configure LangSmith.
LangSmith will help us trace, monitor and debug LangChain applications.
You can sign up for LangSmith [here](https://smith.langchain.com/).
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you DO NOT already have a Milvus Vector Store you want to connect to, see `Milvus Setup` section below before proceeding.

If you DO have a Milvus Vector Store you want to connect to, edit the connection details in `nvidia_rag_canonical/chain.py`

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/nvidia-rag-canonical/playground](http://127.0.0.1:8000/nvidia-rag-canonical/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/nvidia-rag-canonical")
```


## Milvus Setup

Use this step if you need to create a Milvus Vector Store and ingest data.
We will first follow the standard Milvus setup instructions [here](https://milvus.io/docs/install_standalone-docker.md).

1. Download the Docker Compose YAML file.
    ```shell
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml
    ```
2. Start the Milvus Vector Store container
    ```shell
    sudo docker compose up -d
    ```
3. Install the PyMilvus package to interact with the Milvus container.
    ```shell
    pip install pymilvus
    ```
4. Let's now ingest some data! We can do that by moving into this directory and running the code in `ingest.py`, eg:

    ```shell
    python ingest.py
    ```

    Note that you can (and should!) change this to ingest data of your choice.
