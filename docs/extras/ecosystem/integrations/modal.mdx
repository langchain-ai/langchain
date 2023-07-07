# Modal

This page covers how to use the Modal ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Modal wrappers.

## Installation and Setup
- Install with `pip install modal-client`
- Run `modal token new`

## Define your Modal Functions and Webhooks

You must include a prompt. There is a rigid response structure.

```python
class Item(BaseModel):
    prompt: str

@stub.webhook(method="POST")
def my_webhook(item: Item):
    return {"prompt": my_function.call(item.prompt)}
```

An example with GPT2:

```python
from pydantic import BaseModel

import modal

stub = modal.Stub("example-get-started")

volume = modal.SharedVolume().persist("gpt2_model_vol")
CACHE_PATH = "/root/model_cache"

@stub.function(
    gpu="any",
    image=modal.Image.debian_slim().pip_install(
        "tokenizers", "transformers", "torch", "accelerate"
    ),
    shared_volumes={CACHE_PATH: volume},
    retries=3,
)
def run_gpt2(text: str):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    encoded_input = tokenizer(text, return_tensors='pt').input_ids
    output = model.generate(encoded_input, max_length=50, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

class Item(BaseModel):
    prompt: str

@stub.webhook(method="POST")
def get_text(item: Item):
    return {"prompt": run_gpt2.call(item.prompt)}
```

## Wrappers

### LLM

There exists an Modal LLM wrapper, which you can access with 
```python
from langchain.llms import Modal
```