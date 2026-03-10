<div align="center">
  <a href="https://www.langchain.com/">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <img alt="LangChain Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h2>langchain-mistralai</h2>
  <p>Official LangChain integration for Mistral AI — chat models, embeddings, structured output, and tool calling.</p>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/pypi/l/langchain-mistralai" alt="License: MIT"></a>
  <a href="https://pypi.org/project/langchain-mistralai/"><img src="https://img.shields.io/pypi/v/langchain-mistralai?label=version" alt="PyPI version"></a>
  <a href="https://pypi.org/project/langchain-mistralai/"><img src="https://img.shields.io/pypi/pyversions/langchain-mistralai" alt="Python versions"></a>
  <a href="https://pypistats.org/packages/langchain-mistralai"><img src="https://img.shields.io/pepy/dt/langchain-mistralai" alt="Downloads"></a>
</div>

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Chat Model](#chat-model)
  - [Basic Usage](#basic-usage)
  - [Streaming](#streaming)
  - [Structured Output](#structured-output)
  - [Tool Calling](#tool-calling)
- [Embeddings](#embeddings)
- [Retry & Concurrency](#retry--concurrency)
- [Configuration Reference](#configuration-reference)
- [Development](#development)

---

## Installation

```bash
pip install langchain-mistralai
```

Set your API key:

```bash
export MISTRAL_API_KEY="your-api-key"
```

Or pass it directly when initializing the model (see [Configuration Reference](#configuration-reference)).

---

## Quick Start

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest")
response = llm.invoke("What is the capital of France?")
print(response.content)
# -> "The capital of France is Paris."
```

---

## Chat Model

### Basic Usage

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantum entanglement in one sentence."),
]

response = llm.invoke(messages)
print(response.content)
```

### Streaming

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest")

for chunk in llm.stream("Write a haiku about the ocean."):
    print(chunk.content, end="", flush=True)
```

Async streaming:

```python
async for chunk in llm.astream("Write a haiku about the ocean."):
    print(chunk.content, end="", flush=True)
```

### Structured Output

Use `with_structured_output` to get responses that conform to a schema. Supported methods: `function_calling`, `json_mode`, `json_schema`.

```python
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI


class BookInfo(BaseModel):
    title: str
    author: str
    year: int


llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
structured_llm = llm.with_structured_output(BookInfo)

result = structured_llm.invoke("Tell me about '1984' by George Orwell.")
print(result.title)   # -> "1984"
print(result.author)  # -> "George Orwell"
print(result.year)    # -> 1949
```

### Tool Calling

```python
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 25°C."


llm = ChatMistralAI(model="mistral-large-latest")
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in Paris?")
print(response.tool_calls)
```

---

## Embeddings

```python
from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(model="mistral-embed")

# Embed a single query
vector = embeddings.embed_query("What is LangChain?")
print(len(vector))  # -> 1024

# Embed multiple documents
vectors = embeddings.embed_documents([
    "LangChain is a framework for LLM applications.",
    "Mistral AI builds frontier language models.",
])
print(len(vectors))     # -> 2
print(len(vectors[0]))  # -> 1024
```

---

## Retry & Concurrency

`ChatMistralAI` has built-in retry and concurrency controls to make your application resilient under load and rate limits.

### Retry (`max_retries`)

Failed requests are automatically retried using **tenacity** with exponential back-off. The following error types trigger a retry:

| Trigger | Description |
|---|---|
| `httpx.RequestError` | Network-level errors (connection reset, DNS failure, etc.) |
| `httpx.StreamError` | Errors during streaming |
| HTTP `429` | Too Many Requests (rate limited) |
| HTTP `500` | Internal Server Error |
| HTTP `502` | Bad Gateway |
| HTTP `503` | Service Unavailable |
| HTTP `504` | Gateway Timeout |

Non-retryable errors (e.g. `400 Bad Request`, `401 Unauthorized`, `404 Not Found`) are raised immediately without retrying.

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-large-latest",
    max_retries=5,   # default is 5
)
```

### Max Concurrency (`max_concurrent_requests`)

When making many async calls simultaneously (e.g. inside `asyncio.gather`), use `max_concurrent_requests` to cap how many requests are in-flight at once. This helps avoid hitting rate limits.

```python
import asyncio
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-large-latest",
    max_concurrent_requests=10,  # default is 64
)

# At most 10 requests will be in-flight at the same time
responses = await asyncio.gather(*[
    llm.ainvoke(f"Question {i}") for i in range(50)
])
```

### Combined Example

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-large-latest",
    max_retries=5,               # retry up to 5 times on transient failures
    max_concurrent_requests=10,  # cap async concurrency at 10
    timeout=60,                  # per-request timeout in seconds
)
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` / `model_name` | `str` | `"mistral-small"` | Model identifier (e.g. `"mistral-large-latest"`) |
| `api_key` / `mistral_api_key` | `SecretStr` | `$MISTRAL_API_KEY` | Your Mistral API key |
| `base_url` / `endpoint` | `str` | `"https://api.mistral.ai/v1"` | Override the API base URL |
| `temperature` | `float` | `0.7` | Sampling temperature, must be in `[0.0, 1.0]` |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in the response |
| `top_p` | `float` | `1.0` | Nucleus sampling probability, must be in `[0.0, 1.0]` |
| `max_retries` | `int` | `5` | Retry budget for transient failures |
| `max_concurrent_requests` | `int` | `64` | Max simultaneous async requests |
| `timeout` | `int` | `120` | Per-request HTTP timeout in seconds |
| `random_seed` | `int \| None` | `None` | Seed for deterministic sampling |
| `safe_mode` | `bool \| None` | `None` | Enable Mistral safe-prompt injection |
| `streaming` | `bool` | `False` | Enable streaming by default |

---

## Development

### Setup

```bash
# Clone the repo
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/partners/mistralai

# Install with uv
uv sync --group test
```

### Running Tests

Unit tests (no API key required):

```bash
pytest tests/unit_tests/ -v
```

Integration tests (requires `MISTRAL_API_KEY`):

```bash
pytest tests/integration_tests/ -v
```

### Project Structure

```
langchain-mistralai/
├── pyproject.toml
├── langchain_mistralai/
│   ├── __init__.py
│   ├── chat_models.py      # ChatMistralAI — chat completions
│   └── embeddings.py       # MistralAIEmbeddings
└── tests/
    ├── unit_tests/
    │   ├── test_chat_models.py
    │   ├── test_embeddings.py
    │   ├── test_imports.py
    │   ├── test_standard.py
    │   └── test_retry_concurrency.py   # retry & concurrency tests
    └── integration_tests/
        ├── test_chat_models.py
        ├── test_embeddings.py
        └── test_standard.py
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Type Checking

```bash
uv run mypy langchain_mistralai/
```

---

## Related Links

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [LangChain Documentation](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
- [API Reference](https://reference.langchain.com/python/integrations/langchain_mistralai/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Report an Issue](https://github.com/langchain-ai/langchain/issues)

---

<div align="center">
  <sub>Built with ❤️ by the LangChain community.</sub>
</div>
