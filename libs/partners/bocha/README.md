# langchain-bocha

This package contains the LangChain integration with Bocha. Bocha provides powerful web search capabilities and OpenAI-compatible Model APIs (e.g., DeepSeek-V4).

## Quick Install

```bash
pip install langchain-bocha
```

## 🤔 What is this?

This package contains the LangChain integration with Bocha APIs, enabling two core capabilities:
1. **Search Capabilities**: Retrieve real-time search results via `BochaAPIWrapper`, `BochaSearchRun`, and `BochaSearchResults` tools.
2. **Model Capabilities**: Interact with Bocha's Vantage model service (e.g., DeepSeek-V4-Pro) via `ChatBocha`.

## 🚀 Basic Usage

### Using ChatBocha Model

```python
from langchain_bocha import ChatBocha

model = ChatBocha(
    model="deepseek-v4-pro",
    api_key="your-bocha-api-key",
)

response = model.invoke("Hello, who are you?")
print(response.content)
```

### Using Bocha Search Tool

```python
from langchain_bocha import BochaSearchResults

tool = BochaSearchResults(api_key="your-bocha-api-key")
results = tool.invoke("What is the latest news about LangChain?")
print(results)
```

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_bocha/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/bocha).
