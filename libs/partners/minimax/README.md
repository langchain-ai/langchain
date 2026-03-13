# langchain-minimax

This package contains the LangChain integration with [MiniMax](https://www.minimax.io/).

## Quick Install

```bash
pip install langchain-minimax
```

## What is this?

This package provides access to MiniMax's large language models, including
**MiniMax-M2.5**, through LangChain's standard chat model interface. MiniMax
models are accessed via an OpenAI-compatible API.

## Usage

```python
from langchain_minimax import ChatMiniMax

model = ChatMiniMax(
    model="MiniMax-M2.5",
    # api_key="...",  # or set MINIMAX_API_KEY env variable
)

response = model.invoke("Hello, how are you?")
print(response.content)
```

### Streaming

```python
for chunk in model.stream("Tell me a joke"):
    print(chunk.text, end="")
```

### Tool calling

```python
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(
        ..., description="The city and state, e.g. San Francisco, CA"
    )


model_with_tools = model.bind_tools([GetWeather])
ai_msg = model_with_tools.invoke("What is the weather in Beijing?")
print(ai_msg.tool_calls)
```

## Environment Variables

- `MINIMAX_API_KEY`: Your MiniMax API key (required)
- `MINIMAX_API_BASE`: Custom API base URL (default: `https://api.minimax.io/v1`)

## Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and
[Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open
to contributions, whether it be in the form of a new feature, improved
infrastructure, or better documentation.

For detailed information on how to contribute, see the
[Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
