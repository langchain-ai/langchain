# langchain-astraflow

This package contains the LangChain integration with [Astraflow](https://umodelverse.ai) by UCloud (优刻得), an OpenAI-compatible AI model aggregation platform supporting 200+ models.

## Installation

```bash
pip install -U langchain-astraflow
```

## Setup

Set your API key as an environment variable:

```bash
# Global endpoint
export ASTRAFLOW_API_KEY="your-api-key"

# China endpoint
export ASTRAFLOW_CN_API_KEY="your-cn-api-key"
```

## Usage

```python
from langchain_astraflow import ChatAstraflow

# Global endpoint (default)
model = ChatAstraflow(
    model="gpt-4o",
    temperature=0,
    # api_key="...",   # or ASTRAFLOW_API_KEY env var
)

ai_msg = model.invoke("Tell me a joke about programming.")
print(ai_msg.content)

# China endpoint
model_cn = ChatAstraflow(
    model="deepseek-v3",
    base_url="https://api.modelverse.cn/v1",
    api_key="...",  # or ASTRAFLOW_CN_API_KEY env var
)
```

## Supported endpoints

| Region | Base URL | Env var |
|--------|----------|---------|
| Global | `https://api-us-ca.umodelverse.ai/v1` | `ASTRAFLOW_API_KEY` |
| China  | `https://api.modelverse.cn/v1`         | `ASTRAFLOW_CN_API_KEY` |

See [Astraflow documentation](https://umodelverse.ai) for the full list of supported models.
