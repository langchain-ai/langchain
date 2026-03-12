# langchain-minimax

This package contains the LangChain integration with [MiniMax](https://www.minimax.io/).

## Installation

```bash
pip install -U langchain-minimax
```

## Chat Models

`ChatMiniMax` class exposes chat models from MiniMax.

```python
from langchain_minimax import ChatMiniMax

llm = ChatMiniMax(model="MiniMax-M2.5")
llm.invoke("Hello, how are you?")
```

### Environment Setup

Set your MiniMax API key:

```bash
export MINIMAX_API_KEY="your-api-key"
```

### Supported Models

| Model | Context Window | Description |
|---|---|---|
| `MiniMax-M2.5` | 204K tokens | Latest flagship model with reasoning |
| `MiniMax-M2.5-highspeed` | 204K tokens | High-speed variant of M2.5 |
| `MiniMax-M2.1` | 204K tokens | Previous generation model |
| `MiniMax-M2` | 196K tokens | Stable production model |
| `MiniMax-M1` | 1M tokens | Extended context model |
