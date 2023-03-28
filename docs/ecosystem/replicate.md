# Replicate
This page covers how to run models on Replicate within LangChain.

## Installation and Setup
- Create a [Replicate](https://replicate.com) account. Get your API key and set it as an environment variable (`REPLICATE_API_TOKEN`)
- Install the [Replicate python client](https://github.com/replicate/replicate-python) with `pip install replicate`

## Calling a model

Find a model on the [Replicate explore page](https://replicate.com/explore), and then paste in the model name and version in this format: `owner-name/model-name:version`

For example, for this [flan-t5 model](https://replicate.com/daanelson/flan-t5), click on the API tab. The model name/version would be: `daanelson/flan-t5:04e422a9b85baed86a4f24981d7f9953e20c5fd82f6103b74ebc431588e1cec8`

Only the `model` param is required, but any other model parameters can also be passed in with the format `input={model_param: value, ...}`


For example, if we were running stable diffusion and wanted to change the image dimensions:

```
Replicate(model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", input={'image_dimensions': '512x512'})
```

*Note that only the first output of a model will be returned.*
From here, we can initialize our model:

```python
llm = Replicate(model="daanelson/flan-t5:04e422a9b85baed86a4f24981d7f9953e20c5fd82f6103b74ebc431588e1cec8")
```

And run it:

```python
prompt = """
Answer the following yes/no question by reasoning step by step.
Can a dog drive a car?
"""
llm(prompt)
```

We can call any Replicate model (not just LLMs) using this syntax. For example, we can call [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion):

```python
text2image = Replicate(model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                       input={'image_dimensions'='512x512'}

image_output = text2image("A cat riding a motorcycle by Picasso")
```
