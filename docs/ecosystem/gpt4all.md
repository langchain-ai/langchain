# GPT4All

This page covers how to use the `GPT4All` wrapper within LangChain.
It is broken into two parts: installation and setup, and then usage with an example.

## Installation and Setup
- Install the Python package with `pip install pyllamacpp`
- Download a [GPT4All model](https://github.com/nomic-ai/gpt4all) and place it in your desired directory

## Usage

### GPT4All

To use the GPT4All wrapper, you need to provide the path to the pre-trained model file and the model's configuration.
```python
from langchain.llms import GPT4All

# Instantiate the model
model = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)

# Generate text
response = model("Once upon a time, ")
```

You can also customize the generation parameters, such as n_predict, temp, top_p, top_k, and others.

Example:

```python
model = GPT4All(model="./models/gpt4all-model.bin", n_predict=55, temp=0)
response = model("Once upon a time, ")
```
## Model File

You can find links to model file downloads at the [GPT4all](https://github.com/nomic-ai/gpt4all) repository. They will need to be converted to `ggml` format to work, as specified in the [pyllamacpp](https://github.com/nomic-ai/pyllamacpp) repository.

For a more detailed walkthrough of this, see [this notebook](../modules/models/llms/integrations/gpt4all.ipynb)