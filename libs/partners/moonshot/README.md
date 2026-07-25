# langchain-moonshot

This package contains the LangChain integration with [Moonshot AI](https://moonshot.cn).

## Installation

```bash
pip install -U langchain-moonshot
```

## Chat Models

`ChatMoonshot` exposes Moonshot AI's large language models through LangChain's
`BaseChatModel` interface.

```python
from langchain_moonshot import ChatMoonshot

model = ChatMoonshot(model="moonshot-v1-8k")
model.invoke("Hello!")
```

### Tool calling

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


model_with_tools = model.bind_tools([GetWeather])
```

## Environment Variables

- `MOONSHOT_API_KEY` — Moonshot API key (required)
- `MOONSHOT_API_BASE` — Custom API base URL (optional, defaults to `https://api.moonshot.cn/v1`)

For more information see the [Moonshot AI API documentation](https://platform.moonshot.cn/docs).
