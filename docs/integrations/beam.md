# Beam

>[Beam](https://docs.beam.cloud/introduction) makes it easy to run code on GPUs, deploy scalable web APIs, 
> schedule cron jobs, and run massively parallel workloads — without managing any infrastructure.
 

## Installation and Setup

- [Create an account](https://www.beam.cloud/)
- Install the Beam CLI with `curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh`
- Register API keys with `beam configure`
- Set environment variables (`BEAM_CLIENT_ID`) and (`BEAM_CLIENT_SECRET`)
- Install the Beam SDK:
```bash
pip install beam-sdk
```

## LLM


```python
from langchain.llms.beam import Beam
```

### Example of the Beam app

This is the environment you’ll be developing against once you start the app.
It's also used to define the maximum response length from the model.
```python
llm = Beam(model_name="gpt2",
           name="langchain-gpt2-test",
           cpu=8,
           memory="32Gi",
           gpu="A10G",
           python_version="python3.8",
           python_packages=[
               "diffusers[torch]>=0.10",
               "transformers",
               "torch",
               "pillow",
               "accelerate",
               "safetensors",
               "xformers",],
           max_length="50",
           verbose=False)
```

### Deploy the Beam app

Once defined, you can deploy your Beam app by calling your model's `_deploy()` method.

```python
llm._deploy()
```

### Call the Beam app

Once a beam model is deployed, it can be called by calling your model's `_call()` method.
This returns the GPT2 text response to your prompt.

```python
response = llm._call("Running machine learning on a remote GPU")
```

An example script which deploys the model and calls it would be:

```python
from langchain.llms.beam import Beam
import time

llm = Beam(model_name="gpt2",
           name="langchain-gpt2-test",
           cpu=8,
           memory="32Gi",
           gpu="A10G",
           python_version="python3.8",
           python_packages=[
               "diffusers[torch]>=0.10",
               "transformers",
               "torch",
               "pillow",
               "accelerate",
               "safetensors",
               "xformers",],
           max_length="50",
           verbose=False)

llm._deploy()

response = llm._call("Running machine learning on a remote GPU")

print(response)
```