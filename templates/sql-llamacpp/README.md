
# sql-llamacpp

This template enables a user to interact with a SQL database using natural language. 

It uses [Mistral-7b](https://mistral.ai/news/announcing-mistral-7b/) via [llama.cpp](https://github.com/ggerganov/llama.cpp) to run inference locally on a Mac laptop.

## Environment Setup

To set up the environment, use the following steps:

```shell
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
conda create -n llama python=3.9.16
conda activate /Users/rlm/miniforge3/envs/llama
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package sql-llamacpp
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add sql-llamacpp
```

And add the following code to your `server.py` file:
```python
from sql_llamacpp import chain as sql_llamacpp_chain

add_routes(app, sql_llamacpp_chain, path="/sql-llamacpp")
```

The package will download the Mistral-7b model from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF). You can select other files and specify their download path (browse [here](https://huggingface.co/TheBloke)).

This package includes an example DB of 2023 NBA rosters. You can see instructions to build this DB [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).

(Optional) Configure LangSmith for tracing, monitoring and debugging LangChain applications. You can sign up for LangSmith [here](https://smith.langchain.com/). If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

You can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
You can access the playground at [http://127.0.0.1:8000/sql-llamacpp/playground](http://127.0.0.1:8000/sql-llamacpp/playground)  

You can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/sql-llamacpp")
```
