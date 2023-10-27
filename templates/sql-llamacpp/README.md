# SQL with LLaMA2 using Ollama

This template allows you to chat with a SQL database in natural language in private, using an open source LLM.

## LLama.cpp
 
### Enviorment

From [here](https://python.langchain.com/docs/guides/local_llms) and [here](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md).

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
conda create -n llama python=3.9.16
conda activate /Users/rlm/miniforge3/envs/llama
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```

### LLM

It will download Mistral-7b model from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).

You can select other files and specify their download path (browse [here](https://huggingface.co/TheBloke)).

## Set up SQL DB

This template includes an example DB of 2023 NBA rosters.

You can see instructions to build this DB [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).

## Installation

```bash
# from inside your LangServe instance
poe add sql/llama2-ollama
```
