# RAG Example on Intel Gaudi
This template performs RAG using Chroma and Text Generation Inference on Habana Gaudi2. The Intel Gaudi 2 accelerator supports both training and inference for deep learning models in particular for LLMs. Please visit [Habana AI products](https://habana.ai/products) for more details.

## Environment Setup
To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2, please follow these steps:

### Build the Docker image located in the tgi-gaudi repo:
```bash
git clone https://github.com/huggingface/tgi-gaudi.git
cd tgi-gaudi/
docker build -t tgi_gaudi .
```

### Launch a local server instance on 1 Gaudi card:
```bash
model=Intel/neural-chat-7b-v3-3
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model
```

For gated models such as `LLAMA-2`, you will have to pass -e HUGGING_FACE_HUB_TOKEN=\<token\> to the docker run command above with a valid Hugging Face Hub read token.

Please follow this link [huggingface token](https://huggingface.co/docs/hub/security-tokens) to get the access token ans export `HUGGINGFACEHUB_API_TOKEN` environment with the token.

```bash
export HUGGINGFACEHUB_API_TOKEN=<token> 
```

### Launch a local server instance on 8 Gaudi cards:

```bash
model=Intel/neural-chat-7b-v3-3
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model --sharded true --num-shard 8
```

Send a request to check if the endpoint is wokring:

```bash
curl localhost:8080/generate -X POST -d '{"inputs":"Which NFL team won the Super Bowl in the 2010 season?","parameters":{"max_new_tokens":128, "do_sample": true}}'   -H 'Content-Type: application/json'
```
The first call will be slower as the model is compiled.

More details please refer to [tgi-gaudi](https://github.com/huggingface/tgi-gaudi/blob/v1.2-release/README.md).


## Populating with data

If you want to populate the DB with some example data, you can run the below commands:
```shell
poetry install
poetry run python ingest.py
```

The script process and stores sections from Edgar 10k filings data for Nike `nke-10k-2023.pdf` into a Chroma database.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package intel-rag-gaudi
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add intel-rag-gaudi
```

And add the following code to your `server.py` file:
```python
from intel_rag_gaudi import chain as gaudi_rag_chain

add_routes(app, gaudi_rag_chain, path="/intel-rag-gaudi")
```

(Optional) Let's now configure LangSmith. LangSmith will help us trace, monitor and debug LangChain applications. LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). If you don't have access, you can skip this section

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
We can access the playground at [http://127.0.0.1:8000/intel-rag-gaudi/playground](http://127.0.0.1:8000/intel-rag-gaudi/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/intel-rag-gaudi")
```

