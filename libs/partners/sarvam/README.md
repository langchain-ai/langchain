# langchain-sarvam

An integration package connecting [Sarvam AI](https://sarvam.ai) and [LangChain](https://python.langchain.com).

Sarvam AI offers multilingual large language models with native support for all 22 scheduled
Indian languages, advanced reasoning capabilities, and an OpenAI-compatible chat API.

## Features

- `ChatSarvam` — a `BaseChatModel` wrapper for the Sarvam chat completion API
- Synchronous and async generation (`invoke`, `ainvoke`)
- Streaming support (`stream`, `astream`)
- `reasoning_effort` parameter for hybrid thinking mode (`low`, `medium`, `high`)
- Full Indian language support (Hindi, Bengali, Tamil, Telugu, Kannada, …)
- Token usage metadata attached to every response

## Available Models

| Model | Context | Notes |
|---|---|---|
| `sarvam-m` | — | Legacy 24B model, still supported |
| `sarvam-30b` | 64 K tokens | Recommended for general tasks |
| `sarvam-105b` | 128 K tokens | Best quality; complex reasoning & coding |

## Installation

```bash
pip install langchain-sarvam
```

## Setup

Create an API key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) and export it:

```bash
export SARVAM_API_KEY="your-api-key"
```

## Quickstart

```python
from langchain_sarvam import ChatSarvam

model = ChatSarvam(model="sarvam-m")

# Simple invocation
response = model.invoke("What is the capital of India?")
print(response.content)
# → New Delhi
```

### With system message

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Reply only in Hindi."),
    HumanMessage(content="भारत की राजधानी क्या है?"),
]

response = model.invoke(messages)
print(response.content)
```

### Streaming

```python
for chunk in model.stream("Explain the Indian monsoon briefly."):
    print(chunk.content, end="", flush=True)
```

### Async

```python
import asyncio
from langchain_sarvam import ChatSarvam

async def main():
    model = ChatSarvam(model="sarvam-30b")
    response = await model.ainvoke("Name three Indian classical dances.")
    print(response.content)

asyncio.run(main())
```

### Async streaming

```python
async def stream():
    model = ChatSarvam(model="sarvam-30b")
    async for chunk in model.astream("Tell me about the Taj Mahal."):
        print(chunk.content, end="", flush=True)

asyncio.run(stream())
```

### Reasoning mode

```python
model = ChatSarvam(
    model="sarvam-105b",
    reasoning_effort="high",  # "low" | "medium" | "high"
)

response = model.invoke("Solve: If 2x + 3 = 11, what is x?")
print(response.content)
```

### Usage metadata

Every `AIMessage` response includes token counts:

```python
response = model.invoke("Hello!")
print(response.usage_metadata)
# {'input_tokens': 5, 'output_tokens': 12, 'total_tokens': 17}
```

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"sarvam-m"` | Model name |
| `temperature` | `float` | `0.7` | Sampling temperature (0–2) |
| `max_tokens` | `int \| None` | `None` | Max tokens to generate |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability |
| `reasoning_effort` | `str \| None` | `None` | `"low"`, `"medium"`, or `"high"` |
| `streaming` | `bool` | `False` | Enable streaming by default |
| `stop_sequences` | `list[str] \| str \| None` | `None` | Stop sequences |
| `frequency_penalty` | `float \| None` | `None` | Frequency penalty (−2 to 2) |
| `presence_penalty` | `float \| None` | `None` | Presence penalty (−2 to 2) |
| `api_key` | `str \| None` | env `SARVAM_API_KEY` | Sarvam API key |
| `base_url` | `str \| None` | env `SARVAM_API_BASE` | Custom API base URL |
| `model_kwargs` | `dict` | `{}` | Extra parameters forwarded to the API |

## Development

```bash
# Install dependencies (requires uv)
uv sync --group test

# Run unit tests (no API key needed)
make test

# Run integration tests (requires SARVAM_API_KEY)
make integration_tests

# Lint
make lint
```

## Links

- [Sarvam AI documentation](https://docs.sarvam.ai)
- [Sarvam AI dashboard](https://dashboard.sarvam.ai)
- [sarvamai Python SDK](https://pypi.org/project/sarvamai/)
- [LangChain documentation](https://python.langchain.com)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
