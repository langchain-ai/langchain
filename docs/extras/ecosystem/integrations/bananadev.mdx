# Banana

This page covers how to use the Banana ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Banana wrappers.

## Installation and Setup

- Install with `pip install banana-dev`
- Get an Banana api key and set it as an environment variable (`BANANA_API_KEY`)

## Define your Banana Template

If you want to use an available language model template you can find one [here](https://app.banana.dev/templates/conceptofmind/serverless-template-palmyra-base).
This template uses the Palmyra-Base model by [Writer](https://writer.com/product/api/).
You can check out an example Banana repository [here](https://github.com/conceptofmind/serverless-template-palmyra-base).

## Build the Banana app

Banana Apps must include the "output" key in the return json. 
There is a rigid response structure.

```python
# Return the results as a dictionary
result = {'output': result}
```

An example inference function would be:

```python
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        temperature=0.9,
        early_stopping=True,
        no_repeat_ngram_size=3,
        num_beams=5,
        length_penalty=1.5,
        repetition_penalty=1.5,
        bad_words_ids=[[tokenizer.encode(' ', add_prefix_space=True)[0]]]
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return the results as a dictionary
    result = {'output': result}
    return result
```

You can find a full example of a Banana app [here](https://github.com/conceptofmind/serverless-template-palmyra-base/blob/main/app.py).

## Wrappers

### LLM

There exists an Banana LLM wrapper, which you can access with

```python
from langchain.llms import Banana
```

You need to provide a model key located in the dashboard:

```python
llm = Banana(model_key="YOUR_MODEL_KEY")
```
