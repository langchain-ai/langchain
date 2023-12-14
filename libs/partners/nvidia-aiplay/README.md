# langchain-nvidia-aiplay

The `langchain-nvidia-aiplay` package contains LangChain integrations for chat models and embeddings powered by the NVIDIA AI Playground.

>[NVIDIA AI Playground](https://www.nvidia.com/en-us/research/ai-playground/) gives users easy access to hosted endpoints for generative AI models like Llama-2, SteerLM, Mistral, etc. Using the API, you can query NVCR (NVIDIA Container Registry) function endpoints and get quick results from a DGX-hosted cloud compute environment. All models are source-accessible and can be deployed on your own compute cluster.

Below is an example on how to use some common chat model functionality.

## Installation


```python
%pip install -U --quiet langchain-nvidia-aiplay
```

## Setup

**To get started:**
1. Create a free account with the [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) service, which hosts AI solution catalogs, containers, models, etc.
2. Navigate to `Catalog > AI Foundation Models > (Model with API endpoint)`.
3. Select the `API` option and click `Generate Key`.
4. Save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.


```python
import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA AIPLAY API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key
```


```python
## Core LC Chat Interface
from langchain_nvidia_aiplay import ChatNVAIPlay

llm = ChatNVAIPlay(model="mixtral_8x7b")
result = llm.invoke("Write a ballad about LangChain.")
print(result.content)
```


## Stream, Batch, and Async

These models natively support streaming, and as is the case with all LangChain LLMs they expose a batch method to handle concurrent requests, as well as async methods for invoke, stream, and batch. Below are a few examples.


```python
print(llm.batch(["What's 2*3?", "What's 2*6?"]))
# Or via the async API
# await llm.abatch(["What's 2*3?", "What's 2*6?"])
```


```python
for chunk in llm.stream("How far can a seagull fly in one day?"):
    # Show the token separations
    print(chunk.content, end="|")
```


```python
async for chunk in llm.astream("How long does it take for monarch butterflies to migrate?"):
    print(chunk.content, end="|")
```

## Supported models

Querying `available_models` will still give you all of the other models offered by your API credentials.

The `playground_` prefix is optional.


```python
list(llm.available_models)


# ['playground_llama2_13b',
# 'playground_llama2_code_13b',
# 'playground_clip',
# 'playground_fuyu_8b',
# 'playground_mistral_7b',
# 'playground_nvolveqa_40k',
# 'playground_yi_34b',
# 'playground_nemotron_steerlm_8b',
# 'playground_nv_llama2_rlhf_70b',
# 'playground_llama2_code_34b',
# 'playground_mixtral_8x7b',
# 'playground_neva_22b',
# 'playground_steerlm_llama_70b',
# 'playground_nemotron_qa_8b',
# 'playground_sdxl']
```


## Model types

All of these models above are supported and can be accessed via `ChatNVAIPlay`. 

Some model types support unique prompting techniques and chat messages. We will review a few important ones below.


**To find out more about a specific model, please navigate to the API section of an AI Playground model [as linked here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/codellama-13b/api).**

### General Chat

Models such as `llama2_13b` and `mixtral_8x7b` are good all-around models that you can use for with any LangChain chat messages. Example below.


```python
from langchain_nvidia_aiplay import ChatNVAIPlay
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named Fred."),
        ("user", "{input}")
    ]
)
chain = (
    prompt
    | ChatNVAIPlay(model="llama2_13b")
    | StrOutputParser()
)

for txt in chain.stream({"input": "What's your name?"}):
    print(txt, end="")
```


### Code Generation

These models accept the same arguments and input structure as regular chat models, but they tend to perform better on code-genreation and structured code tasks. An example of this is `llama2_code_13b`.


```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert coding AI. Respond only in valid python; no narration whatsoever."),
        ("user", "{input}")
    ]
)
chain = (
    prompt
    | ChatNVAIPlay(model="llama2_code_13b")
    | StrOutputParser()
)

for txt in chain.stream({"input": "How do I solve this fizz buzz problem?"}):
    print(txt, end="")
```

## Steering LLMs

> [SteerLM-optimized models](https://developer.nvidia.com/blog/announcing-steerlm-a-simple-and-practical-technique-to-customize-llms-during-inference/) supports "dynamic steering" of model outputs at inference time.

This lets you "control" the complexity, verbosity, and creativity of the model via integer labels on a scale from 0 to 9. Under the hood, these are passed as a special type of assistant message to the model.

The "steer" models support this type of input, such as `steerlm_llama_70b`


```python
from langchain_nvidia_aiplay import ChatNVAIPlay

llm = ChatNVAIPlay(model="steerlm_llama_70b")
# Try making it uncreative and not verbose
complex_result = llm.invoke(
    "What's a PB&J?",
    labels={"creativity": 0, "complexity": 3, "verbosity": 0}
)
print("Un-creative\n")
print(complex_result.content)

# Try making it very creative and verbose
print("\n\nCreative\n")
creative_result = llm.invoke(
    "What's a PB&J?",
    labels={"creativity": 9, "complexity": 3, "verbosity": 9}
)
print(creative_result.content)
```


#### Use within LCEL

The labels are passed as invocation params. You can `bind` these to the LLM using the `bind` method on the LLM to include it within a declarative, functional chain. Below is an example.


```python
from langchain_nvidia_aiplay import ChatNVAIPlay
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named Fred."),
        ("user", "{input}")
    ]
)
chain = (
    prompt
    | ChatNVAIPlay(model="steerlm_llama_70b").bind(labels={"creativity": 9, "complexity": 0, "verbosity": 9})
    | StrOutputParser()
)

for txt in chain.stream({"input": "Why is a PB&J?"}):
    print(txt, end="")
```

## Multimodal

NVidia also supports multimodal inputs, meaning you can provide both images and text for the model to reason over.

These models also accept `labels`, similar to the Steering LLMs above. In addition to `creativity`, `complexity`, and `verbosity`, these models support a `quality` toggle.

An example model supporting multimodal inputs is `playground_neva_22b`.

These models accept LangChain's standard image formats. Below are examples.


```python
import requests

image_url = "https://picsum.photos/seed/kitten/300/200"
image_content = requests.get(image_url).content
```

Initialize the model like so:

```python
from langchain_nvidia_aiplay import ChatNVAIPlay

llm = ChatNVAIPlay(model="playground_neva_22b")
```

#### Passing an image as a URL


```python
from langchain_core.messages import HumanMessage

llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
    ])
```


```python
### You can specify the labels for steering here as well.  You can try setting a low verbosity, for instance

from langchain_core.messages import HumanMessage

llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
    ],
    labels={
        "creativity": 0,
        "quality": 9,
        "complexity": 0,
        "verbosity": 0
    }
)
```



#### Passing an image as a base64 encoded string


```python
import base64
b64_string = base64.b64encode(image_content).decode('utf-8')
llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}},
        ])
    ])
```

#### Directly within the string

The NVIDIA API uniquely accepts images as base64 images inlined within <img> HTML tags. While this isn't interoperable with other LLMs, you can directly prompt the model accordingly.


```python
base64_with_mime_type = f"data:image/png;base64,{b64_string}"
llm.invoke(
    f'What\'s in this image?\n<img src="{base64_with_mime_type}" />'
)
```



## RAG: Context models

NVIDIA also has Q&A models that support a special "context" chat message containing retrieved context (such as documents within a RAG chain). This is useful to avoid prompt-injecting the model.

**Note:** Only "user" (human) and "context" chat messages are supported for these models, not system or AI messages useful in conversational flows.

The `_qa_` models like `nemotron_qa_8b` support this.


```python
from langchain_nvidia_aiplay import ChatNVAIPlay
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
prompt = ChatPromptTemplate.from_messages(
    [
        ChatMessage(role="context", content="Parrots and Cats have signed the peace accord."),
        ("user", "{input}")
    ]
)
llm = ChatNVAIPlay(model="nemotron_qa_8b")
chain = (
    prompt
    | llm
    | StrOutputParser()
)
chain.invoke({"input": "What was signed?"})
```

## Embeddings

You can also connect to embeddings models through this package. Below is an example:

```
from langchain_nvidia_aiplay import NVAIPlayEmbeddings

embedder = NVAIPlayEmbeddings(model="nvolveqa_40k")
embedder.embed_query("What's the temperature today?")
embedder.embed_documents([
    "The temperature is 42 degrees.",
    "Class is dismissed at 9 PM."
])
```

By default the embedding model will use the "passage" type for documents and "query" type for queries, but you can fix this on the instance.

```python
query_embedder = NVAIPlayEmbeddings(model="nvolveqa_40k", model_type="query")
doc_embeddder = NVAIPlayEmbeddings(model="nvolveqa_40k", model_type="passage")
```

