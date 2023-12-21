# langchain-mistralai

This package contains the LangChain integrations for [MistralAI](https://docs.mistral.ai) through their [mistralai](https://pypi.org/project/mistralai/) SDK.

## Installation

```bash
pip install -U langchain-mistralai
```

## Chat Models

This package contains the `ChatMistralAI` class, which is the recommended way to interface with MistralAI models.

To use, install the requirements, and configure your environment.

```bash
export MISTRAL_API_KEY=your-api-key
```

Then initialize

```python
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

chat = ChatMistralAI(model="mistral-small")
messages = [HumanMessage(content="say a brief hello")]
chat.invoke(messages)
```

`ChatMistralAI` also supports async and streaming functionality:

```python
# For async...
await chat.ainvoke(messages)

# For streaming...
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```
