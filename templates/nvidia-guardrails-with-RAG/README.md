# Guardrails Integration with LangChain Expression Language Template

This example demonstrates integrating NeMo Guardrails with LangChain Expression Language Templates. Here we start of with a `nvidia-rag-canonical` from the LangChain [templates](https://github.com/langchain-ai/langchain/tree/master/templates/nvidia-rag-canonical) which performs RAG using Milvus Vector Store and NVIDIA Models, both embedding and Chat.

## Environmental Setup

### Exporting API keys as Environment Variables

To start working with this example, you should start with exporting NVIDIA API Key and NGC API Key as environment variables. To create the NVIDIA API Key, you can create one by following these steps:

1. Create a free account with the NVIDIA GPU Cloud service, which hosts AI solution catalogs, containers, models, etc.

2. Navigate to `Catalog > AI Foundation Models >  (Model with API endpoint).`

3. Select the `API` option and click `Generate Key`.

4. Save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.

To create the NGC API Key, you can create one by following these steps:

1. create a free account with the [NVIDIA NGC Catalog](https://ngc.nvidia.com/signin). 

2. Navigate to `Setup > Generate API Key` and follow the steps.

3. Save the generated key as `NGC_API_KEY`. From there, you should have access to NGC models

```
export NVIDIA_API_KEY=<your nvidia_api_key>
export NGC_API_KEY=<your ngc_api_key>
```
### NeMo Guardrails

To add guardrails, first you need to clone the nemoguardrails package with langchain runnable integrations as follows:

Create a directory named `nemoguardrails`, clone and install the package with all necessary dependencies

```
mkdir nemoguardrails
cd nemoguardrails
git clone https://github.com/NVIDIA/NeMo-Guardrails.git
cd NeMo-Guardrails
pip install -e .
```

### Milvus Setup
This template uses a RAG use case with Milvis vector database. Before running the app, make sure to set up the vector database Milvis using the following steps and ingest any necessary data for future retrieval. 

1. Download the Docker Compose YAML file

```
wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

2. Start the Milvus Vectorstore container using the following command.

```
sudo docker compose up -d
```

3. Install the PyMilvus package to interact with the Milvus container.

```
pip install pymilvus
```

4. To ingest data, we can do that by moving into the directory with `ingest.py` file and running the code.

```
python ingest.py
```

## Usage
To run this example, you should first have the LangChain CLI installed:
```
pip install -U langchain-cli
```

To use the NVIDIA models, install the langchain NVIDIA AI Endpoints package:
```
pip install -U langchain_nvidia_aiplay
```

To create a new LangChain project and install this as the only package, you can do:
```
langchain app new my-app --package nvidia-guardrails-with-RAG
```

If you want to add this to an existing project, you can just run:
```
langchain app add nvidia-guardrails-with-RAG
```

And add the following code to your `server.py` file:
```python
from nvidia_guardrails_with_RAG import chain_with_guardrails as nvidia_guardrails_with_RAG_chain

add_routes(app, nvidia_guardrails_with_RAG_chain, path="/nvidia-guardrails-with-RAG")
```

If you want to set up an ingestion pipeline, you can add the following code to your `server.py` file:
```python
from nvidia_guardrails_with_RAG import ingest as nvidia_guardrails_ingest

add_routes(app, nvidia_guardrails_ingest, path="/nvidia-rag-ingest")
```
Note that for files ingested by the ingestion API, the server will need to be restarted for the newly ingested files to be accessible by the retriever.

If you DO NOT already have a Milvus Vector Store you want to connect to, see `Milvus Setup` section below before proceeding.

If you DO have a Milvus Vector Store you want to connect to, edit the connection details in `nvidia_rag_canonical/chain.py`

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

## Guardrails integration

To add guardrails, create a guardrails directory in the working directory (with the `chain_with_guardrails.py` script) and add the necessary config. Currently, rails added are self check input/output rails, fact checking rails. You can add as many rails as you wish. To understand how to do this, refer to the [Guardrails Documentation](https://github.com/NVIDIA/NeMo-Guardrails)

To add the guardrails, start with creating a config which consists of `config.yml`, `prompts.yml`, any other topical rails of choice (optional)

## Sample Input + Output

### RAG example

"Question": "How many Americans receive Social Security Benefits?"
"Answer": "According to the Social Security Administration, about 65 million Americans receive Social Security benefits."

