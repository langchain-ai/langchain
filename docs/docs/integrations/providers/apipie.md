# APIpie

>[APIpie](https://apipie.ai/) is a unified API gateway that provides access to multiple LLM providers through a single API.

## Installation and Setup

```bash
pip install langchain-community
```

You'll need to set the `APIPIE_API_KEY` environment variable to your APIpie API key.

```python
import os
os.environ["APIPIE_API_KEY"] = "your-api-key"
```

Alternatively, you can pass your API key directly when initializing the chat model:

```python
from langchain_community.chat_models import ChatAPIpie
chat = ChatAPIpie(apipie_api_key="your-api-key")
```

## Chat Models

The `ChatAPIpie` class provides access to APIpie's chat models. It supports various models from different providers through a unified interface.

```python
from langchain_community.chat_models import ChatAPIpie
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatAPIpie(model="openai/gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = chat.invoke(messages)
print(response.content)
```

### Environment Variables

The following environment variables can be used to configure the `ChatAPIpie` class:

- `APIPIE_API_KEY`: Your APIpie API key
- `APIPIE_API_BASE`: Base URL for API requests (default: "https://apipie.ai/v1")
- `APIPIE_PROXY`: HTTP proxy to use for API requests

### Available Models

You can retrieve the list of available models from APIpie:

```python
# Get available models
available_models = ChatAPIpie.get_available_models(apipie_api_key="your-api-key")
print(available_models)
```

### Model Parameters

The `ChatAPIpie` class supports a wide range of parameters to customize the behavior of the language model:

```python
from langchain_community.chat_models import ChatAPIpie
from langchain_core.messages import HumanMessage

# Initialize with various parameters
chat = ChatAPIpie(
    model="openai/gpt-4o",
    temperature=0.5,
    max_tokens=500,
    streaming=True
)

# Use the model
response = chat.invoke([HumanMessage(content="Tell me a joke about programming.")])
print(response.content)
```

#### Basic Parameters

- `model` (default: "openai/gpt-4o"): Language model to use
- `temperature` (default: 0.7): Controls randomness in the output (0.0 to 2.0)
- `top_p` (default: 1.0): Controls diversity via nucleus sampling (0.0 to 1.0)
- `top_k` (default: 0): Limits token selection to top k most probable tokens
- `frequency_penalty` (default: 0.0): Penalizes tokens based on their frequency (-2.0 to 2.0)
- `presence_penalty` (default: 0.0): Penalizes tokens that have already appeared (-2.0 to 2.0)
- `repetition_penalty` (default: 1.0): Discourages repeating the same words or phrases (1.0 to 2.0)
- `beam_size` (default: 1): Number of sequences to keep at each step of generation
- `max_tokens` (default: None): Maximum number of tokens to generate
- `n` (default: 1): Number of chat completions to generate for each prompt
- `streaming` (default: False): Whether to stream the results

#### Memory Parameters

- `memory` (default: False): Enable integrated model memory to maintain conversation context
- `mem_session` (default: None): Unique identifier for maintaining separate memory chains
- `mem_expire` (default: None): Time in minutes after which stored memories will expire
- `mem_clear` (default: 0): Set to 1 to instantly delete all stored memories for the specified session
- `mem_msgs` (default: 8): Maximum number of messages to append from memory
- `mem_length` (default: 20): Percentage of model's max response tokens to use for memory

#### RAG and Routing Parameters

- `rag_tune` (default: None): Name of the RAG tune or vector collection for RAG tuning
- `routing` (default: "perf"): Define how to route calls when multiple providers exist

#### Tool Parameters

- `tools` (default: None): List of tools to integrate into the chat model
- `tool_choice` (default: "none"): Specifies how the tools are chosen
- `tools_model` (default: "gpt-4o-mini"): Model to use for processing tools

#### Integrity Parameters

- `integrity` (default: None): Integrity setting (12 or 13) for querying and returning best answers
- `integrity_model` (default: "gpt-4o"): Model to use for integrity checks
- `force_provider` (default: False): Force request to be routed to the specified provider

### Streaming

You can use the streaming interface to get tokens as they're generated:

```python
from langchain_community.chat_models import ChatAPIpie
from langchain_core.messages import HumanMessage

chat = ChatAPIpie(model="openai/gpt-4o", streaming=True)

for chunk in chat.stream([HumanMessage(content="Tell me a joke about programming.")]):
    print(chunk.content, end="", flush=True)
```

### Async Usage

You can also use the async interface for non-blocking calls:

```python
import asyncio
from langchain_community.chat_models import ChatAPIpie
from langchain_core.messages import HumanMessage

async def main():
    chat = ChatAPIpie(model="openai/gpt-4o")
    response = await chat.ainvoke([HumanMessage(content="Hello, how are you?")])
    print(response.content)

asyncio.run(main())
```

### Async Streaming

```python
import asyncio
from langchain_community.chat_models import ChatAPIpie
from langchain_core.messages import HumanMessage

async def main():
    chat = ChatAPIpie(model="openai/gpt-4o", streaming=True)
    async for chunk in chat.astream([HumanMessage(content="Hello, how are you?")]):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```
