# langchain-deepseek

This package contains the LangChain integration with the DeepSeek API

## Installation

```bash
pip install -U langchain-deepseek
```

And you should configure credentials by setting the following environment variables:

* `DEEPSEEK_API_KEY`

## Chat Models

`ChatDeepSeek` class exposes chat models from DeepSeek.

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat")
llm.invoke("Sing a ballad of LangChain.")
```
