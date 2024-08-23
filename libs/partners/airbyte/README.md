# langchain-airbyte

This package contains the LangChain integration with Airbyte

## Installation

```bash
pip install -U langchain-airbyte
```

The integration package doesn't have any global environment variables that need to be
set, but some integrations (e.g. `source-github`) may need credentials passed in.

## Document Loaders

`AirbyteLoader` class exposes a single document loader for Airbyte sources.

```python
from langchain_airbyte import AirbyteLoader

loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 100},
)
docs = loader.load()
```
